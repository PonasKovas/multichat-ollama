mod config;
mod tls;

use chrono::{DateTime, Utc};
use chrono_humanize::{Accuracy, HumanTime, Tense};
use clap::Parser;
use config::Config;
use multichat_client::proto::Config as ProtoConfig;
use multichat_client::{ClientBuilder, UpdateKind};
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::path::PathBuf;
use std::time::Duration;
use tokio::fs;
use tokio::time::timeout;
use tracing::{error, info, subscriber};
use tracing_subscriber::filter::{EnvFilter, LevelFilter};
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;

#[derive(Parser)]
struct Args {
    #[clap(help = "Path to config file")]
    config: PathBuf,
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<MessageObject>,
    stream: bool,
    keep_alive: String,
    options: OllamaOptions,
}
#[derive(Serialize)]
struct MessageObject {
    role: String,
    content: String,
}
#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    top_k: u32,
}

#[derive(Debug)]
struct MessageMemory {
    was_llm: bool,
    time: DateTime<Utc>,
    message: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    let registry = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().without_time().with_target(false));

    subscriber::set_global_default(registry).unwrap();

    let args = Args::parse();

    info!("Reading config from {}", args.config.display());

    let config = match fs::read_to_string(&args.config).await {
        Ok(config) => config,
        Err(err) => {
            return Err(format!("Error reading config: {}", err).into());
        }
    };

    let config = match toml::from_str::<Config>(&config) {
        Ok(config) => config,
        Err(err) => {
            return Err(format!("Error parsing config: {}", err).into());
        }
    };

    // group name -> memories
    let mut all_memories: HashMap<String, Vec<String>> =
        serde_json::from_str(&fs::read_to_string(&config.ollama.memory_file).await?)?;

    let connector = match config.multichat.certificate {
        Some(certificate) => match tls::configure(&certificate).await {
            Ok(connector) => Some(connector),
            Err(err) => {
                return Err(format!("Error configuring TLS: {}", err).into());
            }
        },
        None => None,
    };

    let mut proto_config = ProtoConfig::default();
    proto_config.max_size(512 * 1024 * 1024); // 512 MiB

    let (groups, mut client) = match timeout(
        Duration::from_secs(5),
        ClientBuilder::maybe_tls(connector)
            .config(proto_config)
            .connect(&config.multichat.server, config.multichat.access_token),
    )
    .await?
    {
        Ok((groups, client)) => (groups, client),
        Err(err) => {
            return Err(format!("Error connecting to multichat: {}", err).into());
        }
    };

    info!("Connected to Multichat");

    let reqw_client = reqwest::Client::new();

    let mut message_histories: HashMap<u32, VecDeque<MessageMemory>> = HashMap::new();
    let mut usernames: HashMap<(u32, u32), String> = HashMap::new();
    let mut my_uid: Vec<(u32, u32)> = Vec::new();

    for group in &config.multichat.groups {
        let gid = *groups.get(group.as_str()).ok_or("Group not found")?;
        client.join_group(gid).await?;

        let uid = client.join_user(gid, &config.multichat.user_name).await?;
        my_uid.push((gid, uid));
        usernames.insert((gid, uid), config.multichat.user_name.clone());
        message_histories.insert(gid, VecDeque::new());
    }

    loop {
        let update = client.read_update().await?;

        match update.kind {
            UpdateKind::Join(username) | UpdateKind::Rename(username) => {
                usernames.insert((update.gid, update.uid), username.clone());
            }
            UpdateKind::Leave => {
                usernames.remove(&(update.gid, update.uid));
            }
            UpdateKind::Message(message) => {
                let memories = all_memories
                    .entry(
                        groups
                            .iter()
                            .find(|(_, id)| **id == update.gid)
                            .unwrap()
                            .0
                            .clone()
                            .into_owned(),
                    )
                    .or_insert_with(Vec::new);

                // add the message to the chat history
                let message_history = message_histories
                    .get_mut(&update.gid)
                    .expect("received message from unknown group");
                if message_history.len() == config.ollama.prompt_messages_n {
                    message_history.pop_front();
                }

                let was_llm = my_uid.contains(&(update.gid, update.uid));

                let memory = MessageMemory {
                    was_llm,
                    time: Utc::now(),
                    message: format!(
                        "{name}: {msg}",
                        name = usernames.get(&(update.gid, update.uid)).unwrap(),
                        msg = message.message,
                    ),
                };

                info!(
                    "(msg history: {}): {:?}",
                    message_history.len(),
                    memory.message
                );

                message_history.push_back(memory);

                // the following code only applies for messages from other users
                if was_llm {
                    continue;
                }

                let my_uid = my_uid
                    .iter()
                    .find(|(gid, _uid)| *gid == update.gid)
                    .expect("message from group where bot is not present")
                    .1;

                // handle some commands
                if message.message.trim().starts_with("/memories")
                    || message.message.trim().starts_with("/mems")
                {
                    client
                        .send_message(
                            update.gid,
                            my_uid,
                            &memories
                                .iter()
                                .enumerate()
                                .map(|(i, m)| format!("{i} - {m}\n"))
                                .collect::<String>(),
                            &[],
                        )
                        .await?;
                    continue;
                }
                if message.message.trim().starts_with("/rmem")
                    || message.message.trim().starts_with("/rmemory")
                {
                    if let Some(idx) = message.message.trim().split_whitespace().nth(1) {
                        match idx.parse::<usize>() {
                            Err(e) => {
                                client
                                    .send_message(update.gid, my_uid, &format!("{e:?}"), &[])
                                    .await?
                            }
                            Ok(idx) => {
                                if idx >= memories.len() {
                                    client
                                        .send_message(
                                            update.gid,
                                            my_uid,
                                            "invalid id, use /mems to list",
                                            &[],
                                        )
                                        .await?
                                } else {
                                    let memory = memories.remove(idx);
                                    client
                                        .send_message(
                                            update.gid,
                                            my_uid,
                                            &format!("removed {memory:?}"),
                                            &[],
                                        )
                                        .await?;
                                    // save
                                    fs::write(
                                        &config.ollama.memory_file,
                                        &serde_json::to_string_pretty(&all_memories)?,
                                    )
                                    .await?;
                                }
                            }
                        }
                    } else {
                        client
                            .send_message(
                                update.gid,
                                my_uid,
                                "/rmem <index> - remove a memory (/mems to list)",
                                &[],
                            )
                            .await?;
                    }
                    continue;
                }

                // check if this new message mentions the bot
                if !is_substring_isolated(&message.message, &config.ollama.mention_name) {
                    continue;
                }

                let system_prompt = config
                    .ollama
                    .system_prompt
                    .replace("{mention_name}", &config.ollama.mention_name)
                    .replace(
                        "{memories}",
                        &memories
                            .iter()
                            .map(|m| format!("- {m}\n"))
                            .collect::<String>(),
                    );
                let mut messages: Vec<_> = vec![
                    MessageObject {
                        role: "system".to_string(),
                        content: system_prompt,
                    },
                    MessageObject {
                        role: "assistant".to_string(),
                        content: format!("Hello everyone! I'm back! Ready to fulfill your questionable requests and be obedient! :)"),
                    },
                ];

                messages.extend(message_history.iter().map(|msg| {
                    info!("{:?}", format_memory(&msg));
                    MessageObject {
                        role: if msg.was_llm { "assistant" } else { "user" }.to_string(),
                        content: format_memory(&msg),
                    }
                }));
                let body = OllamaRequest {
                    model: config.ollama.model.clone(),
                    messages,
                    stream: false,
                    keep_alive: "30s".to_string(), // how long to keep the model loaded for
                    options: OllamaOptions {
                        temperature: config.ollama.temperature,
                        top_k: config.ollama.top_k,
                    },
                };

                let mut url = config.ollama.base_url.clone();
                url.set_path("api/chat");
                let response = reqw_client
                    .post(url)
                    .basic_auth(
                        &config.ollama.basic_auth_user,
                        Some(&config.ollama.basic_auth_password),
                    )
                    .json(&body)
                    .send()
                    .await;

                match response {
                    Ok(res) => {
                        if !res.status().is_success() {
                            error!("Failed request. Status: {}. {res:?}", res.status());
                            client
                                .send_message(
                                    update.gid,
                                    my_uid,
                                    &format!("Failed ollama request. Status: {}", res.status()),
                                    &[],
                                )
                                .await?;
                            continue;
                        }

                        // get the generated message
                        let json = res.json::<Value>().await?;
                        let msg = json["message"]["content"].as_str().unwrap();

                        // check if new memory created
                        if let Some(memory) = extract_between_tags(msg, "<MEMORY>", "</MEMORY>") {
                            memories.push(memory.to_owned());
                            // save
                            fs::write(
                                &config.ollama.memory_file,
                                &serde_json::to_string_pretty(&all_memories)?,
                            )
                            .await?;
                        }

                        // reply with the message contents
                        for msg in msg.split("\n\n") {
                            let cleaned_msg = remove_quotes(
                                remove_prefix_case_insensitive(
                                    remove_quotes(msg.trim()).trim(),
                                    "ollama: ",
                                )
                                .trim(),
                            )
                            .trim();

                            if cleaned_msg.is_empty() {
                                continue;
                            }

                            client
                                .send_message(update.gid, my_uid, cleaned_msg, &[])
                                .await?;
                        }
                    }
                    Err(e) => eprintln!("Request error: {:?}", e),
                }
            }
        }
    }
}

fn is_substring_isolated(s: &str, substr: &str) -> bool {
    if let Some(index) = s.to_lowercase().find(substr.to_lowercase().as_str()) {
        // Check the character before the substring
        let before_is_valid = if index == 0 {
            true // Nothing before, valid
        } else {
            // Find the character just before the substring
            !s[..index].chars().rev().next().unwrap().is_alphabetic()
        };

        // Check the character after the substring
        let after_is_valid = if index + substr.len() == s.len() {
            true // Nothing after, valid
        } else {
            // Find the character just after the substring
            !s[(index + substr.len())..]
                .chars()
                .next()
                .unwrap()
                .is_alphabetic()
        };

        before_is_valid && after_is_valid
    } else {
        false
    }
}

fn remove_prefix_case_insensitive<'a>(s: &'a str, prefix: &'a str) -> &'a str {
    if s.to_lowercase().starts_with(&prefix.to_lowercase()) {
        &s[prefix.len()..]
    } else {
        s
    }
}

fn remove_quotes(s: &str) -> &str {
    if let Some(stripped) = s.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        stripped
    } else {
        s
    }
}

fn format_memory(memory: &MessageMemory) -> String {
    if memory.was_llm {
        format!("{}", memory.message)
    } else {
        format!(
            "{} {}",
            HumanTime::from(memory.time).to_text_en(Accuracy::Rough, Tense::Past),
            memory.message
        )
    }
}

fn extract_between_tags<'a>(
    text: &'a str,
    start_tag: &'a str,
    end_tag: &'a str,
) -> Option<&'a str> {
    if let Some(start_idx) = text.find(start_tag) {
        let start = start_idx + start_tag.len();
        if let Some(end_idx) = text[start..].find(end_tag) {
            return Some(&text[start..start + end_idx]);
        }
    }
    None
}

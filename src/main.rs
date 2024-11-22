mod config;
mod tls;

use clap::Parser;
use config::Config;
use multichat_client::proto::Config as ProtoConfig;
use multichat_client::{ClientBuilder, UpdateKind};
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::path::PathBuf;
use tokio::fs;
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

    let (groups, mut client) = match ClientBuilder::maybe_tls(connector)
        .config(proto_config)
        .connect(&config.multichat.server, config.multichat.access_token)
        .await
    {
        Ok((groups, client)) => (groups, client),
        Err(err) => {
            return Err(format!("Error connecting to multichat: {}", err).into());
        }
    };

    info!("Connected to Multichat");

    let reqw_client = reqwest::Client::new();

    let mut message_histories: HashMap<u32, VecDeque<(u32, String)>> = HashMap::new();
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
                // add the message to the chat history
                let message_history = message_histories
                    .get_mut(&update.gid)
                    .expect("received message from unknown group");
                if message_history.len() == config.ollama.prompt_messages_n {
                    message_history.pop_front();
                }
                message_history.push_back((update.uid, message.message.clone()));

                // the following code only applies for messages from other users
                if my_uid.contains(&(update.gid, update.uid)) {
                    continue;
                }

                // check if this new message mentions the bot
                if !is_substring_isolated(&message.message, &config.ollama.mention_name) {
                    continue;
                }

                let my_uid = my_uid
                    .iter()
                    .find(|(gid, _uid)| *gid == update.gid)
                    .expect("message from group where bot is not present")
                    .1;

                let system_prompt = config
                    .ollama
                    .system_prompt
                    .replace("{mention_name}", &config.ollama.mention_name);
                let mut messages: Vec<_> = vec![MessageObject {
                    role: "system".to_string(),
                    content: system_prompt,
                }];

                messages.extend(message_history.iter().map(|(uid, msg)| {
                    if *uid == my_uid {
                        MessageObject {
                            role: "assistant".to_string(),
                            content: format!("{}", msg),
                        }
                    } else {
                        let name = usernames.get(&(update.gid, *uid)).unwrap().clone();

                        MessageObject {
                            role: "user".to_string(),
                            content: format!("{}: {}", name, msg),
                        }
                    }
                }));
                let body = OllamaRequest {
                    model: config.ollama.model.clone(),
                    messages,
                    stream: false,
                    keep_alive: "5m".to_string(), // keep the model loaded for 5 minutes after this request
                    options: OllamaOptions {
                        temperature: config.ollama.temperature,
                    },
                };

                let mut url = config.ollama.base_url.clone();
                url.set_path("api/chat");
                let response = reqw_client
                    .post(url)
                    .basic_auth("mykola", Some("5M9dOmGQldSJONaCSIaVmZRUF"))
                    .json(&body)
                    .send()
                    .await;

                match response {
                    Ok(res) => {
                        if !res.status().is_success() {
                            error!("Failed request. Status: {}", res.status());
                            continue;
                        }

                        // get the generated message
                        let json = res.json::<Value>().await?;
                        let msg = json["message"]["content"].as_str().unwrap();

                        // reply with the message contents
                        info!("sending reply: {msg:?}");
                        client.send_message(update.gid, my_uid, msg, &[]).await?;
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

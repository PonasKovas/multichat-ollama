use crate::{
    ollama_api::{OllamaRequest, OllamaRequestMessage, OllamaRequestOptions, OllamaResponse},
    room_state::Message,
    State,
};
use anyhow::Context;
use multichat_client::{Update, UpdateKind};
use tokio::task::JoinHandle;

pub async fn handle_mc_update(state: &mut State, update: Update) -> anyhow::Result<()> {
    // some convenience macros
    macro_rules! room {
        () => {
            state
                .rooms
                .get_mut(&update.gid)
                .context("received update for group im not in")?
        };
    }
    macro_rules! send {
        ($msg:expr) => {
            state
                .mc_client
                .send_message(update.gid, room!().my_uid, $msg, &[])
        };
    }

    match update.kind {
        UpdateKind::Join(username) | UpdateKind::Rename(username) => {
            room!().usernames.insert(update.uid, username.clone());
        }
        UpdateKind::Leave => {
            room!().usernames.remove(&update.uid);
        }
        UpdateKind::Message(message) => {
            if room!().my_uid == update.uid {
                // dont care about my own messages
                return Ok(());
            }

            let mut image = None;
            for attachment in &message.attachments {
                let bytes = state.mc_client.download_attachment(attachment.id).await?;
                // only save image types
                let is_image = match bytes.as_slice() {
                    [0xFF, 0xD8, 0xFF, ..]
                    | [0x89, b'P', b'N', b'G', ..]
                    | [0x52, 0x49, 0x46, 0x46, ..] => true,
                    _ => false,
                };
                if is_image {
                    image = Some(bytes);
                    break;
                }
            }

            state.push_message(update.gid, Message::new(&message.message, false, image));

            // handle some commands
            let trimmed = message.message.trim();
            if trimmed.starts_with("/memories") || trimmed.starts_with("/mems") {
                let formatted_mems = room!()
                    .memories
                    .iter()
                    .enumerate()
                    .map(|(i, m)| format!("{i} - {m}\n"))
                    .collect::<String>();

                send!(&formatted_mems).await?;

                return Ok(());
            }
            if trimmed.starts_with("/rmem") || trimmed.starts_with("/rmemory") {
                if let Some(idx) = message.message.trim().split_whitespace().nth(1) {
                    match idx.parse::<usize>() {
                        Err(e) => {
                            send!(&format!("{e:?}")).await?;
                        }
                        Ok(idx) => {
                            if idx >= room!().memories.len() {
                                send!("invalid id, use /mems to list").await?;
                            } else {
                                let memory = state.remove_memory(update.gid, idx).await?;

                                send!(&format!("removed {memory:?}")).await?;
                            }
                        }
                    }
                } else {
                    send!("/rmem <index> - remove a memory (/mems to list)").await?;
                }

                return Ok(());
            }

            // check if this new message mentions the bot
            if is_substring_isolated(&message.message, &state.config.ollama.mention_name) {
                // if this message mentions the bot, generate a response
                room!().ollama_api_task = Some(send_ollama_request(state, update.gid)?);
            } else {
                // if it doesnt mention the bot, but the bot is currently generating a response, start generating
                // it again with the new message
                if room!().ollama_api_task.is_some() {
                    room!().ollama_api_task = Some(send_ollama_request(state, update.gid)?);
                }
            }
        }
    }

    Ok(())
}

fn send_ollama_request(
    state: &State,
    gid: u32,
) -> anyhow::Result<JoinHandle<anyhow::Result<String>>> {
    macro_rules! room {
        () => {
            state
                .rooms
                .get(&gid)
                .context("received update for group im not in")?
        };
    }

    // Prepare the ollama API request
    let system_prompt = state
        .config
        .ollama
        .system_prompt
        .replace("{mention_name}", &state.config.ollama.mention_name)
        .replace(
            "{memories}",
            &room!()
                .memories
                .iter()
                .map(|m| format!("- {m}\n"))
                .collect::<String>(),
        );
    let mut messages: Vec<_> = vec![
        OllamaRequestMessage {
            role: "system".to_string(),
            content: system_prompt,
            images: Vec::new(),
        },
        OllamaRequestMessage {
            role: "assistant".to_string(),
            content: format!("Hello everyone! I'm back! Ready to be racist and fun with you! :)"),
            images: Vec::new(),
        },
    ];

    // add the real message history
    // we only keep the last image, so find out which message has it
    let last_image_idx = room!()
        .message_history
        .iter()
        .rev()
        .position(|msg| msg.image.is_some());
    messages.extend(
        room!()
            .message_history
            .iter()
            .rev() // double reverse because we have the last_image_idx from the end
            .enumerate()
            .map(|(i, msg)| OllamaRequestMessage {
                role: if msg.was_me { "assistant" } else { "user" }.to_string(),
                content: msg.format(),
                images: if last_image_idx.map(|idx| idx == i).unwrap_or(false) {
                    vec![msg.image.clone().unwrap()]
                } else {
                    vec![]
                },
            })
            .rev(),
    );
    let body = OllamaRequest {
        model: state.config.ollama.model.clone(),
        messages,
        stream: false,
        keep_alive: "30s".to_string(), // how long to keep the model loaded for
        options: OllamaRequestOptions {
            temperature: state.config.ollama.temperature,
            top_k: state.config.ollama.top_k,
        },
    };

    let mut url = state.config.ollama.base_url.clone();
    url.set_path("api/chat");

    let auth_user = state.config.ollama.basic_auth_user.clone();
    let auth_password = state.config.ollama.basic_auth_password.clone();

    let reqw = state.reqw.clone();

    // spawn a task to send a request to the ollama api
    let join_handle = tokio::spawn(async move {
        let response = reqw
            .post(url)
            .basic_auth(&auth_user, Some(&auth_password))
            .json(&body)
            .send()
            .await;

        let response = response?.error_for_status()?;

        let response = response.json::<OllamaResponse>().await?;

        Ok(response.message.content)
    });

    Ok(join_handle)
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

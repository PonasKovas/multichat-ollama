use crate::{room_state::Message, State};
use tracing::error;

pub async fn handle_ollama_gen(
    state: &mut State,
    gid: u32,
    res: anyhow::Result<String>,
) -> anyhow::Result<()> {
    // finished generating response to some chatroom
    state.rooms.get_mut(&gid).unwrap().ollama_api_task = None;
    let my_uid = state.rooms[&gid].my_uid;

    let response = match res {
        Ok(r) => r,
        Err(e) => {
            error!("Failed ollama request. {e:?}");
            state
                .mc_client
                .send_message(gid, my_uid, &format!("Failed ollama request. {e}"), &[])
                .await?;
            return Ok(());
        }
    };

    let response = clean_generated_msg(&response, &state.config.ollama.mention_name);

    state.push_message(gid, Message::new(response, true, None));

    // check if new memory created
    if let Some(memory) = extract_between_tags(response, "<MEMORY>", "</MEMORY>") {
        state.add_memory(gid, memory.to_owned()).await?;
    }

    // reply with the message contents
    for msg in response.split("\n\n") {
        let cleaned_msg = clean_generated_msg(&msg, &state.config.ollama.mention_name);
        if cleaned_msg.is_empty() {
            continue;
        }

        state
            .mc_client
            .send_message(gid, my_uid, cleaned_msg, &[])
            .await?;
    }

    Ok(())
}

fn clean_generated_msg<'a, 'b>(msg: &'a str, llm_name: &'b str) -> &'a str {
    // Trim
    // Remove quotes
    // Trim
    // Remove "ollama: "
    // Trim
    // Remove quotes
    // Trim
    remove_quotes(remove_prefix_case_insensitive(remove_quotes(msg.trim()).trim(), llm_name).trim())
        .trim()
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

fn remove_prefix_case_insensitive<'a, 'b>(s: &'a str, prefix: &'b str) -> &'a str {
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

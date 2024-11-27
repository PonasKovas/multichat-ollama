use base64::Engine;
use chrono::{DateTime, Utc};
use chrono_humanize::{Accuracy, HumanTime, Tense};
use std::collections::{HashMap, VecDeque};
use tokio::task::JoinHandle;

/// State of a particular room/group that ollama is in
pub struct RoomState {
    pub my_uid: u32,
    pub room_name: String,
    pub usernames: HashMap<u32, String>,
    pub message_history: VecDeque<Message>,
    pub memories: Vec<String>,

    pub ollama_api_task: Option<JoinHandle<anyhow::Result<String>>>,
}

#[derive(Debug)]
pub struct Message {
    pub was_me: bool,
    pub time: DateTime<Utc>,
    pub message: String,
    // base64
    pub image: Option<String>,
}

impl RoomState {
    pub fn new(my_uid: u32, room_name: String, memories: Vec<String>) -> Self {
        RoomState {
            my_uid,
            room_name,
            usernames: HashMap::new(),
            message_history: VecDeque::new(),
            memories,
            ollama_api_task: None,
        }
    }
}

impl Message {
    pub fn new(msg: &str, was_me: bool, image: Option<Vec<u8>>) -> Self {
        Message {
            was_me,
            time: Utc::now(),
            message: msg.to_string(),
            image: image.map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes)),
        }
    }
    pub fn format(&self) -> String {
        let timestamp = HumanTime::from(self.time).to_text_en(Accuracy::Rough, Tense::Past);

        if self.was_me {
            format!("{}", self.message)
        } else {
            format!("{} {}", timestamp, self.message)
        }
    }
}

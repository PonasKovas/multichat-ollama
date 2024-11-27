use serde::{Deserialize, Serialize};

// REQUEST
//////////

#[derive(Serialize, Debug)]
pub struct OllamaRequest {
    pub model: String,
    pub messages: Vec<OllamaRequestMessage>,
    pub stream: bool,
    pub keep_alive: String,
    pub options: OllamaRequestOptions,
}

#[derive(Serialize, Debug)]
pub struct OllamaRequestMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct OllamaRequestOptions {
    pub temperature: f32,
    pub top_k: u32,
}

// RESPONSE
///////////

#[derive(Deserialize, Debug)]
pub struct OllamaResponse {
    pub message: OllamaResponseMessage,
}

#[derive(Deserialize, Debug)]
pub struct OllamaResponseMessage {
    pub content: String,
}

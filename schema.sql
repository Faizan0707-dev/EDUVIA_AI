CREATE TABLE users (
    user_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    phone VARCHAR(15) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chats (
    user_id VARCHAR(64) NOT NULL,
    chat_id VARCHAR(64) NOT NULL,
    title VARCHAR(255) NOT NULL DEFAULT 'New Chat',
    quiz_id VARCHAR(64) NULL,
    quiz_score INT NULL,
    quiz_submitted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, chat_id)
);

CREATE TABLE messages (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    chat_id VARCHAR(64) NOT NULL,
    role VARCHAR(20) NOT NULL,
    type VARCHAR(30) NULL,
    content LONGTEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id, chat_id)
        REFERENCES chats (user_id, chat_id)
        ON DELETE CASCADE
);

CREATE TABLE quiz_questions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    chat_id VARCHAR(64) NOT NULL,
    quiz_id VARCHAR(64) NOT NULL,
    question_no INT NOT NULL,
    question TEXT NOT NULL,
    option_1 TEXT NOT NULL,
    option_2 TEXT NOT NULL,
    option_3 TEXT NOT NULL,
    option_4 TEXT NOT NULL,
    correct_answer TEXT NOT NULL,
    selected_answer TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_quiz_question (user_id, chat_id, quiz_id, question_no),
    FOREIGN KEY (user_id, chat_id)
        REFERENCES chats (user_id, chat_id)
        ON DELETE CASCADE
);
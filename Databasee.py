import os
import mysql.connector
from dotenv import load_dotenv
import uuid
import bcrypt

load_dotenv()


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "EDUVIA_AI"),
    )


def save_user_chats(user_id, chats):
    conn = get_connection()
    cur = conn.cursor()

    try:
        for chat_id, chat in chats.items():

            cur.execute(
                "INSERT INTO chats "
                "(user_id, chat_id, title, quiz_id, quiz_score, quiz_submitted) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE title=VALUES(title)",
                (
                    user_id,
                    chat_id,
                    chat.get("title", "New Chat")[:40],
                    None,
                    None,
                    False
                )
            )

            history = chat.get("history", [])

            cur.execute(
                "DELETE FROM messages "
                "WHERE user_id=%s AND chat_id=%s",
                (user_id, chat_id)
            )

            for message in history:

                role = message.get("role", "assistant")
                message_type = message.get("type")

                content = (
                    message.get("quiz_id", "")
                    if message_type == "quiz"
                    else message.get("content", "")
                )

                cur.execute(
                    "INSERT INTO messages "
                    "(user_id, chat_id, role, type, content) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (user_id, chat_id, role, message_type, content)
                )

            for quiz in history:

                if quiz.get("type") != "quiz":
                    continue

                quiz_id = quiz.get("quiz_id")

                if not quiz_id:
                    continue

                for i, q in enumerate(quiz.get("questions", [])):

                    options = q.get("options", [])

                    if len(options) != 4:
                        continue

                    answer = quiz.get("answers", {}).get(str(i))

                    cur.execute(
                        "INSERT INTO quiz_questions "
                        "(user_id, chat_id, quiz_id, question_no, question, "
                        "option_1, option_2, option_3, option_4, "
                        "correct_answer, selected_answer) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                        "ON DUPLICATE KEY UPDATE "
                        "question=VALUES(question), "
                        "option_1=VALUES(option_1), "
                        "option_2=VALUES(option_2), "
                        "option_3=VALUES(option_3), "
                        "option_4=VALUES(option_4), "
                        "correct_answer=VALUES(correct_answer), "
                        "selected_answer=VALUES(selected_answer)",
                        (
                            user_id,
                            chat_id,
                            quiz_id,
                            i + 1,
                            q["question"],
                            options[0],
                            options[1],
                            options[2],
                            options[3],
                            q["answer"],
                            answer
                        )
                    )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()

def delete_user_chat(user_id, chat_id):
    conn = get_connection()
    cur = conn.cursor()

    try:
        for table in ("quiz_questions", "messages", "chats"):
            cur.execute(
                f"DELETE FROM {table} WHERE user_id=%s AND chat_id=%s",
                (user_id, chat_id)
            )
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()


def load_user_chats(user_id):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    try:
        chats = {}

        cur.execute(
            "SELECT chat_id, title FROM chats WHERE user_id=%s",
            (user_id,)
        )

        for row in cur.fetchall():
            chats[row["chat_id"]] = {
                "title": row["title"],
                "history": [],
                "quiz": [],
                "quiz_answers": {},
                "quiz_score": None,
                "quiz_submitted": False,
                "quiz_question_history": []
            }

        cur.execute(
            "SELECT chat_id, role, type, content FROM messages "
            "WHERE user_id=%s ORDER BY id",
            (user_id,)
        )

        for row in cur.fetchall():
            chat = chats.get(row["chat_id"])

            if chat is None:
                continue

            if row["type"] == "quiz":
                chat["history"].append({
                    "role": "quiz",
                    "type": "quiz",
                    "quiz_id": row["content"],
                    "questions": [],
                    "answers": {},
                    "score": None,
                    "submitted": False
                })
            else:
                message = {
                    "role": row["role"],
                    "content": row["content"]
                }

                if row["type"]:
                    message["type"] = row["type"]

                chat["history"].append(message)

        cur.execute(
            "SELECT chat_id, quiz_id, question_no, question, "
            "option_1, option_2, option_3, option_4, "
            "correct_answer, selected_answer "
            "FROM quiz_questions WHERE user_id=%s ORDER BY id",
            (user_id,)
        )

        quiz_data = {}

        for row in cur.fetchall():
            chat = chats.get(row["chat_id"])

            if chat is None:
                continue

            quiz_id = row["quiz_id"]

            if quiz_id not in quiz_data:
                quiz_data[quiz_id] = {
                    "chat_id": row["chat_id"],
                    "questions": [],
                    "answers": {}
                }

            quiz = quiz_data[quiz_id]

            quiz["questions"].append({
                "question": row["question"],
                "options": [
                    row["option_1"],
                    row["option_2"],
                    row["option_3"],
                    row["option_4"]
                ],
                "answer": row["correct_answer"]
            })

            chat["quiz_question_history"].append(row["question"])

            if row["selected_answer"] is not None:
                quiz["answers"][str(row["question_no"] - 1)] = (
                    row["selected_answer"]
                )

        for quiz_id, data in quiz_data.items():
            chat = chats[data["chat_id"]]
            questions = data["questions"]
            answers = data["answers"]

            submitted = (
                len(questions) > 0
                and len(answers) == len(questions)
            )

            score = None

            if submitted:
                score = sum(
                    answers.get(str(i)) == q["answer"]
                    for i, q in enumerate(questions)
                )

            quiz_event = {
                "role": "quiz",
                "type": "quiz",
                "quiz_id": quiz_id,
                "questions": questions,
                "answers": answers,
                "score": score,
                "submitted": submitted
            }

            found = False

            for message in chat["history"]:
                if (
                    message.get("type") == "quiz"
                    and message.get("quiz_id") == quiz_id
                ):
                    message.update(quiz_event)
                    found = True
                    break

            if not found:
                chat["history"].append(quiz_event)

        return chats

    finally:
        cur.close()
        conn.close()
def clean_phone(phone):
    digits = "".join(ch for ch in phone if ch.isdigit())

    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]

    return digits

def create_user(name, email, phone, password):
    name = name.strip()
    email = email.strip().lower()
    phone = clean_phone(phone)

    name_part = email[:-len("@gmail.com")]
    if not (
        email.endswith("@gmail.com")
        and name_part
        and "@" not in name_part
        and " " not in name_part
    ):
        return None, "bad_email"

    if not (len(phone) == 10 and phone[0] in "6789"):
        return None, "bad_phone"

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")
    user_id = uuid.uuid4().hex

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("SELECT 1 FROM users WHERE email=%s", (email,))
        if cur.fetchone():
            return None, "email"

        cur.execute("SELECT 1 FROM users WHERE phone=%s", (phone,))
        if cur.fetchone():
            return None, "phone"

        cur.execute(
            "INSERT INTO users (user_id, name, email, phone, password_hash) "
            "VALUES (%s, %s, %s, %s, %s)",
            (user_id, name, email, phone, password_hash)
        )
        conn.commit()
        return user_id, None

    except mysql.connector.IntegrityError:
        conn.rollback()
        return None, "exists"

    finally:
        cur.close()
        conn.close()
def verify_user(login_id, password):
    login_id = login_id.strip().lower()

    if "@" in login_id:
        column, value = "email", login_id
    else:
        column, value = "phone", clean_phone(login_id)

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            f"SELECT user_id, password_hash FROM users WHERE {column}=%s",
            (value,)
        )
        row = cur.fetchone()

        if row and bcrypt.checkpw(
            password.encode("utf-8"), row[1].encode("utf-8")
        ):
            return row[0]
        return None

    finally:
        cur.close()
        conn.close()
def user_exists(user_id):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("SELECT 1 FROM users WHERE user_id=%s", (user_id,))
        return cur.fetchone() is not None

    finally:
        cur.close()
        conn.close()

def change_password(user_id, old_password, new_password):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("SELECT password_hash FROM users WHERE user_id=%s", (user_id,))
        row = cur.fetchone()

        if not row or not bcrypt.checkpw(
            old_password.encode("utf-8"), row[0].encode("utf-8")
        ):
            return "wrong_old"

        new_hash = bcrypt.hashpw(
            new_password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")
        cur.execute(
            "UPDATE users SET password_hash=%s WHERE user_id=%s",
            (new_hash, user_id)
        )
        conn.commit()
        return None

    finally:
        cur.close()
        conn.close()
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.g2p import G2PService
from app.schemas import (
    LibraryItemCreate,
    LibraryItemResponse,
    LibraryItemUpdate,
    LibraryStatsResponse,
    LibraryWordPreview,
    PracticeAttemptSummary,
    WeakPhonemeStat,
)

SEED_LIBRARY_ITEMS: list[dict[str, str]] = [
    # --- English US (en) Single Words ---
    {
        "text": "Pronunciation",
        "item_type": "word",
        "language": "en",
        "difficulty": "Intermediate",
        "category": "Vocabulary",
        "focus_phoneme": "Stress & /ʃ/",
        "meaning_or_tip": "Stress the 4th syllable: pro-nun-ci-AY-shun (notice the 2nd syllable is 'nun', not 'noun').",
    },
    {
        "text": "Thought",
        "item_type": "word",
        "language": "en",
        "difficulty": "Beginner",
        "category": "Tricky Sounds",
        "focus_phoneme": "/θ/ Unvoiced TH",
        "meaning_or_tip": "Place your tongue tip lightly between your teeth and blow air gently without vibrating your throat.",
    },
    {
        "text": "Weather",
        "item_type": "word",
        "language": "en",
        "difficulty": "Beginner",
        "category": "Daily Essentials",
        "focus_phoneme": "/ð/ Voiced TH",
        "meaning_or_tip": "Round your lips for 'W', then touch your tongue tip between your front teeth and buzz your throat for 'th'.",
    },
    {
        "text": "Comfortable",
        "item_type": "word",
        "language": "en",
        "difficulty": "Intermediate",
        "category": "Daily Essentials",
        "focus_phoneme": "Syllable Reduction",
        "meaning_or_tip": "Native speakers compress this into 3 or 4 quick beats: KUMF-ter-bul, stressing the first syllable.",
    },
    {
        "text": "Schedule",
        "item_type": "word",
        "language": "en",
        "difficulty": "Intermediate",
        "category": "Work & Interviews",
        "focus_phoneme": "/sk/ + /dʒ/",
        "meaning_or_tip": "Start with a crisp 'sk' blend, stress 'SKED', and round your lips slightly for '-jool'.",
    },
    {
        "text": "Entrepreneur",
        "item_type": "word",
        "language": "en",
        "difficulty": "Advanced",
        "category": "Work & Interviews",
        "focus_phoneme": "Final Syllable Stress",
        "meaning_or_tip": "Put the strongest, longest beat on the final syllable: ahn-truh-pruh-NUR.",
    },
    {
        "text": "World",
        "item_type": "word",
        "language": "en",
        "difficulty": "Intermediate",
        "category": "Tricky Sounds",
        "focus_phoneme": "/ɜːld/ R-L Cluster",
        "meaning_or_tip": "Curl your tongue back for 'er', then glide the tip up behind your top front teeth for 'l' before 'd'.",
    },
    {
        "text": "Specific",
        "item_type": "word",
        "language": "en",
        "difficulty": "Intermediate",
        "category": "Work & Interviews",
        "focus_phoneme": "2nd-Syllable Stress",
        "meaning_or_tip": "Keep the first syllable super short ('spuh') and punch the middle syllable: spuh-SIF-ik.",
    },
    # --- English US (en) Phrases & Sentences ---
    {
        "text": "She can see the sea",
        "item_type": "phrase",
        "language": "en",
        "difficulty": "Beginner",
        "category": "Tricky Sounds",
        "focus_phoneme": "/ʃ/ vs /s/",
        "meaning_or_tip": "Round your lips slightly for 'She' (/ʃ/), then pull your lip corners back for 'see' and 'sea' (/s/).",
    },
    {
        "text": "I'll think about it today",
        "item_type": "phrase",
        "language": "en",
        "difficulty": "Beginner",
        "category": "Daily Essentials",
        "focus_phoneme": "Connected Speech ‿",
        "meaning_or_tip": "Link 'think ‿ about ‿ it' smoothly like one flowing word without pausing between them.",
    },
    {
        "text": "The weather is very warm today",
        "item_type": "phrase",
        "language": "en",
        "difficulty": "Beginner",
        "category": "Daily Essentials",
        "focus_phoneme": "/w/ vs /v/",
        "meaning_or_tip": "Touch top teeth to bottom lip for 'very' (/v/), and round lips without teeth for 'weather' & 'warm' (/w/).",
    },
    {
        "text": "Could you help me check this schedule",
        "item_type": "phrase",
        "language": "en",
        "difficulty": "Intermediate",
        "category": "Work & Interviews",
        "focus_phoneme": "Polite Intonation",
        "meaning_or_tip": "Blend 'Could ‿ you' smoothly and stress the content words 'help', 'check', and 'schedule'.",
    },
    {
        "text": "She thought about the pronunciation",
        "item_type": "phrase",
        "language": "en",
        "difficulty": "Intermediate",
        "category": "Tricky Sounds",
        "focus_phoneme": "/θ/ & Word Stress",
        "meaning_or_tip": "Use a soft unvoiced 'th' on 'thought' and stress the '-AY-' syllable in 'pronunciation'.",
    },
    {
        "text": "Three free throws in a row",
        "item_type": "phrase",
        "language": "en",
        "difficulty": "Advanced",
        "category": "Tricky Sounds",
        "focus_phoneme": "/θɹ/ vs /fɹ/",
        "meaning_or_tip": "Switch cleanly between tongue-between-teeth ('three', 'throws') and teeth-on-lip ('free').",
    },
    {
        "text": "We'd love to collaborate on the new project",
        "item_type": "phrase",
        "language": "en",
        "difficulty": "Advanced",
        "category": "Work & Interviews",
        "focus_phoneme": "Business Fluency",
        "meaning_or_tip": "Stress 'love', 'la' in collaborate, and 'pro' in project while linking 'collaborate ‿ on'.",
    },
    # --- English UK (en-gb) ---
    {
        "text": "Water",
        "item_type": "word",
        "language": "en-gb",
        "difficulty": "Beginner",
        "category": "Daily Essentials",
        "focus_phoneme": "Long /ɔː/ Vowel",
        "meaning_or_tip": "Round your lips into an oval for a deep British 'WAW-tuh' with a soft non-rhotic ending.",
    },
    {
        "text": "Can I have a cup of tea please",
        "item_type": "phrase",
        "language": "en-gb",
        "difficulty": "Beginner",
        "category": "Travel & Dining",
        "focus_phoneme": "Weak Forms & Linking",
        "meaning_or_tip": "Reduce 'a' and 'of' to quick schwa sounds ('cup ‿ uh tea') and keep 'tea' long.",
    },
    # --- Spanish (es) ---
    {
        "text": "Gracias",
        "item_type": "word",
        "language": "es",
        "difficulty": "Beginner",
        "category": "Daily Essentials",
        "focus_phoneme": "Pure Vowels /a/",
        "meaning_or_tip": "Thank you — keep every 'a' vowel crisp, open, and uniform without gliding.",
    },
    {
        "text": "Buenos días, ¿cómo estás?",
        "item_type": "phrase",
        "language": "es",
        "difficulty": "Beginner",
        "category": "Daily Essentials",
        "focus_phoneme": "Rising Question Pitch",
        "meaning_or_tip": "Good morning, how are you? — stress 'dí' and 'tás' with a rising question melody.",
    },
    # --- French (fr) ---
    {
        "text": "Bonjour",
        "item_type": "word",
        "language": "fr",
        "difficulty": "Beginner",
        "category": "Daily Essentials",
        "focus_phoneme": "Nasal Vowel & /ʒ/",
        "meaning_or_tip": "Hello — soft 'zh' (/ʒ/) in the second syllable with gentle lip rounding.",
    },
    {
        "text": "Je voudrais un café s'il vous plaît",
        "item_type": "phrase",
        "language": "fr",
        "difficulty": "Beginner",
        "category": "Travel & Dining",
        "focus_phoneme": "Smooth Liaison",
        "meaning_or_tip": "I would like a coffee please — keep syllable timing even and smooth.",
    },
    # --- German (de) ---
    {
        "text": "Entschuldigung",
        "item_type": "word",
        "language": "de",
        "difficulty": "Intermediate",
        "category": "Daily Essentials",
        "focus_phoneme": "/ʃ/ & Cluster",
        "meaning_or_tip": "Excuse me — stress the second syllable ('ent-SHUL-di-gung') with rounded lips for 'sch'.",
    },
    {
        "text": "Guten Morgen, wie geht es Ihnen",
        "item_type": "phrase",
        "language": "de",
        "difficulty": "Beginner",
        "category": "Daily Essentials",
        "focus_phoneme": "Long Vowels",
        "meaning_or_tip": "Good morning, how are you? — hold the 'ee' vowel in 'wie' and 'Ihnen' cleanly.",
    },
    # --- Hindi (hi) ---
    {
        "text": "नमस्ते",
        "item_type": "word",
        "language": "hi",
        "difficulty": "Beginner",
        "category": "Daily Essentials",
        "focus_phoneme": "Dental /त/",
        "meaning_or_tip": "Hello (Namaste) — touch the tongue tip to the back of your upper teeth for a soft dental 't'.",
    },
    {
        "text": "आप कैसे हैं",
        "item_type": "phrase",
        "language": "hi",
        "difficulty": "Beginner",
        "category": "Daily Essentials",
        "focus_phoneme": "Long /आ/ & Diphthong",
        "meaning_or_tip": "How are you? (Aap kaise hain) — open your mouth fully for 'आप' and nasalize 'हैं' gently.",
    },
]


class PronunciationLibraryDB:
    """SQLite persistence layer for words, phrases, IPA/syllable metadata, and learner practice history."""

    def __init__(self, db_path: Path, g2p: G2PService | None = None) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.g2p = g2p
        self._init_schema()
        self._seed_if_empty()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS practice_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    item_type TEXT NOT NULL DEFAULT 'phrase',
                    language TEXT NOT NULL DEFAULT 'en',
                    difficulty TEXT NOT NULL DEFAULT 'Beginner',
                    category TEXT NOT NULL DEFAULT 'Daily Essentials',
                    focus_phoneme TEXT NOT NULL DEFAULT '',
                    meaning_or_tip TEXT NOT NULL DEFAULT '',
                    expected_ipa TEXT NOT NULL DEFAULT '',
                    words_breakdown_json TEXT NOT NULL DEFAULT '[]',
                    best_score INTEGER NOT NULL DEFAULT 0,
                    attempts_count INTEGER NOT NULL DEFAULT 0,
                    last_practiced_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_practice_items_lang
                ON practice_items(language, item_type, difficulty);

                CREATE TABLE IF NOT EXISTS practice_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER,
                    reference_text TEXT NOT NULL,
                    language TEXT NOT NULL,
                    overall_score INTEGER NOT NULL,
                    accuracy_score INTEGER NOT NULL,
                    fluency_score INTEGER NOT NULL,
                    prosody_score INTEGER NOT NULL,
                    weak_phonemes_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def _compute_g2p_metadata(self, text: str, language: str) -> tuple[str, list[dict[str, Any]]]:
        if self.g2p is None:
            return "", []
        try:
            targets = self.g2p.phonemize_text(reference_text=text, language=language)
            full_ipa = " ".join(t.sanitized_ipa for t in targets if t.sanitized_ipa)
            words_breakdown = [
                {
                    "word": t.word,
                    "expected_ipa": t.sanitized_ipa,
                    "phonemes": t.phonemes,
                    "syllables": [
                        {"syllable_ipa": s.ipa, "stress": s.stress} for s in t.syllables
                    ],
                }
                for t in targets
            ]
            return full_ipa, words_breakdown
        except Exception:
            return "", []

    def _seed_if_empty(self) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM practice_items").fetchone()
            if row and int(row["cnt"]) > 0:
                return

            now_iso = datetime.now(UTC).isoformat()
            for item in SEED_LIBRARY_ITEMS:
                full_ipa, breakdown = self._compute_g2p_metadata(item["text"], item["language"])
                conn.execute(
                    """
                    INSERT INTO practice_items (
                        text, item_type, language, difficulty, category,
                        focus_phoneme, meaning_or_tip, expected_ipa,
                        words_breakdown_json, best_score, attempts_count, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?)
                    """,
                    (
                        item["text"],
                        item["item_type"],
                        item["language"],
                        item["difficulty"],
                        item["category"],
                        item["focus_phoneme"],
                        item["meaning_or_tip"],
                        full_ipa,
                        json.dumps(breakdown, ensure_ascii=False),
                        now_iso,
                    ),
                )
            conn.commit()

    def _row_to_item(self, row: sqlite3.Row) -> LibraryItemResponse:
        try:
            raw_words = json.loads(row["words_breakdown_json"] or "[]")
        except Exception:
            raw_words = []
        words = [LibraryWordPreview.model_validate(w) for w in raw_words]
        return LibraryItemResponse(
            id=int(row["id"]),
            text=str(row["text"]),
            item_type="word" if str(row["item_type"]) == "word" else "phrase",
            language=str(row["language"]),
            difficulty=str(row["difficulty"]),
            category=str(row["category"]),
            focus_phoneme=str(row["focus_phoneme"] or ""),
            meaning_or_tip=str(row["meaning_or_tip"] or ""),
            expected_ipa=str(row["expected_ipa"] or ""),
            words_breakdown=words,
            best_score=int(row["best_score"] or 0),
            attempts_count=int(row["attempts_count"] or 0),
            last_practiced_at=row["last_practiced_at"],
            created_at=str(row["created_at"]),
        )

    def list_items(
        self,
        language: str | None = None,
        item_type: str | None = None,
        difficulty: str | None = None,
        category: str | None = None,
        search: str | None = None,
    ) -> list[LibraryItemResponse]:
        clauses: list[str] = []
        params: list[Any] = []
        if language and language != "all":
            clauses.append("language = ?")
            params.append(language)
        if item_type and item_type in ("word", "phrase"):
            clauses.append("item_type = ?")
            params.append(item_type)
        if difficulty and difficulty != "all":
            clauses.append("LOWER(difficulty) = LOWER(?)")
            params.append(difficulty)
        if category and category != "all":
            clauses.append("LOWER(category) = LOWER(?)")
            params.append(category)
        if search and search.strip():
            q = f"%{search.strip()}%"
            clauses.append("(text LIKE ? OR focus_phoneme LIKE ? OR meaning_or_tip LIKE ? OR category LIKE ?)")
            params.extend([q, q, q, q])

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM practice_items {where_sql} ORDER BY id ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_item(r) for r in rows]

    def get_item(self, item_id: int) -> LibraryItemResponse | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM practice_items WHERE id = ?", (item_id,)).fetchone()
        return self._row_to_item(row) if row else None

    def create_item(self, payload: LibraryItemCreate) -> LibraryItemResponse:
        clean_text = payload.text.strip()
        inferred_type = payload.item_type or ("word" if len(clean_text.split()) == 1 else "phrase")
        full_ipa, breakdown = self._compute_g2p_metadata(clean_text, payload.language)
        now_iso = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO practice_items (
                    text, item_type, language, difficulty, category,
                    focus_phoneme, meaning_or_tip, expected_ipa,
                    words_breakdown_json, best_score, attempts_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?)
                """,
                (
                    clean_text,
                    inferred_type,
                    payload.language,
                    payload.difficulty,
                    payload.category,
                    payload.focus_phoneme.strip(),
                    payload.meaning_or_tip.strip(),
                    full_ipa,
                    json.dumps(breakdown, ensure_ascii=False),
                    now_iso,
                ),
            )
            new_id = int(cur.lastrowid or 0)
            conn.commit()
            row = conn.execute("SELECT * FROM practice_items WHERE id = ?", (new_id,)).fetchone()
        assert row is not None
        return self._row_to_item(row)

    def update_item(self, item_id: int, payload: LibraryItemUpdate) -> LibraryItemResponse | None:
        existing = self.get_item(item_id)
        if existing is None:
            return None
        new_text = payload.text.strip() if payload.text is not None else existing.text
        new_lang = payload.language if payload.language is not None else existing.language
        new_type = (
            payload.item_type
            if payload.item_type is not None
            else ("word" if len(new_text.split()) == 1 else "phrase")
        )
        new_diff = payload.difficulty if payload.difficulty is not None else existing.difficulty
        new_cat = payload.category if payload.category is not None else existing.category
        new_focus = payload.focus_phoneme if payload.focus_phoneme is not None else existing.focus_phoneme
        new_tip = payload.meaning_or_tip if payload.meaning_or_tip is not None else existing.meaning_or_tip

        if new_text != existing.text or new_lang != existing.language or not existing.expected_ipa:
            full_ipa, breakdown = self._compute_g2p_metadata(new_text, new_lang)
            breakdown_json = json.dumps(breakdown, ensure_ascii=False)
        else:
            full_ipa = existing.expected_ipa
            breakdown_json = json.dumps(
                [w.model_dump() for w in existing.words_breakdown], ensure_ascii=False
            )

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE practice_items
                SET text = ?, item_type = ?, language = ?, difficulty = ?,
                    category = ?, focus_phoneme = ?, meaning_or_tip = ?,
                    expected_ipa = ?, words_breakdown_json = ?
                WHERE id = ?
                """,
                (
                    new_text,
                    new_type,
                    new_lang,
                    new_diff,
                    new_cat,
                    new_focus,
                    new_tip,
                    full_ipa,
                    breakdown_json,
                    item_id,
                ),
            )
            conn.commit()
        return self.get_item(item_id)

    def delete_item(self, item_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM practice_items WHERE id = ?", (item_id,))
            conn.commit()
            return cur.rowcount > 0

    def record_attempt(
        self,
        reference_text: str,
        language: str,
        overall_score: int,
        accuracy_score: int,
        fluency_score: int,
        prosody_score: int,
        weak_phonemes: list[str],
        item_id: int | None = None,
    ) -> None:
        now_iso = datetime.now(UTC).isoformat()
        clean_ref = reference_text.strip()
        with self._connect() as conn:
            resolved_id = item_id
            if resolved_id is None:
                row = conn.execute(
                    "SELECT id FROM practice_items WHERE LOWER(text) = LOWER(?) AND language = ? LIMIT 1",
                    (clean_ref, language),
                ).fetchone()
                if row:
                    resolved_id = int(row["id"])

            conn.execute(
                """
                INSERT INTO practice_attempts (
                    item_id, reference_text, language, overall_score,
                    accuracy_score, fluency_score, prosody_score,
                    weak_phonemes_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_id,
                    clean_ref,
                    language,
                    int(overall_score),
                    int(accuracy_score),
                    int(fluency_score),
                    int(prosody_score),
                    json.dumps(weak_phonemes, ensure_ascii=False),
                    now_iso,
                ),
            )
            if resolved_id is not None:
                conn.execute(
                    """
                    UPDATE practice_items
                    SET best_score = MAX(best_score, ?),
                        attempts_count = attempts_count + 1,
                        last_practiced_at = ?
                    WHERE id = ?
                    """,
                    (int(overall_score), now_iso, resolved_id),
                )
            conn.commit()

    def get_stats(self, language: str | None = None) -> LibraryStatsResponse:
        with self._connect() as conn:
            if language and language != "all":
                items_rows = conn.execute(
                    "SELECT * FROM practice_items WHERE language = ?", (language,)
                ).fetchall()
                attempts_rows = conn.execute(
                    "SELECT * FROM practice_attempts WHERE language = ? ORDER BY id DESC LIMIT 30",
                    (language,),
                ).fetchall()
            else:
                items_rows = conn.execute("SELECT * FROM practice_items").fetchall()
                attempts_rows = conn.execute(
                    "SELECT * FROM practice_attempts ORDER BY id DESC LIMIT 30"
                ).fetchall()

        total_items = len(items_rows)
        words_count = sum(1 for r in items_rows if r["item_type"] == "word")
        phrases_count = total_items - words_count
        mastered_count = sum(1 for r in items_rows if int(r["best_score"] or 0) >= 85)
        practiced_count = sum(1 for r in items_rows if int(r["attempts_count"] or 0) > 0)

        total_attempts = len(attempts_rows)
        avg_score = (
            int(round(sum(int(r["overall_score"]) for r in attempts_rows) / total_attempts))
            if total_attempts > 0
            else 0
        )

        phoneme_counts: dict[str, int] = {}
        recent_attempts: list[PracticeAttemptSummary] = []
        for r in attempts_rows[:12]:
            try:
                weak_list = json.loads(r["weak_phonemes_json"] or "[]")
            except Exception:
                weak_list = []
            for ph in weak_list:
                if ph:
                    phoneme_counts[str(ph)] = phoneme_counts.get(str(ph), 0) + 1
            recent_attempts.append(
                PracticeAttemptSummary(
                    id=int(r["id"]),
                    item_id=int(r["item_id"]) if r["item_id"] is not None else None,
                    reference_text=str(r["reference_text"]),
                    language=str(r["language"]),
                    overall_score=int(r["overall_score"]),
                    accuracy_score=int(r["accuracy_score"]),
                    fluency_score=int(r["fluency_score"]),
                    prosody_score=int(r["prosody_score"]),
                    weak_phonemes=[str(p) for p in weak_list],
                    created_at=str(r["created_at"]),
                )
            )

        weak_stats = [
            WeakPhonemeStat(phoneme=k, error_count=v)
            for k, v in sorted(phoneme_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        ]

        return LibraryStatsResponse(
            total_items=total_items,
            words_count=words_count,
            phrases_count=phrases_count,
            practiced_count=practiced_count,
            mastered_count=mastered_count,
            total_attempts=total_attempts,
            average_score=avg_score,
            weak_phonemes=weak_stats,
            recent_attempts=recent_attempts,
        )

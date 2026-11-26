export interface LibrarySyllablePreview {
  syllable_ipa: string;
  stress: 'primary' | 'secondary' | 'unstressed';
}

export interface LibraryWordPreview {
  word: string;
  expected_ipa: string;
  phonemes: string[];
  syllables: LibrarySyllablePreview[];
}

export interface LibraryItem {
  id: number;
  text: string;
  item_type: 'word' | 'phrase';
  language: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | string;
  category: string;
  focus_phoneme: string;
  meaning_or_tip: string;
  expected_ipa: string;
  words_breakdown: LibraryWordPreview[];
  best_score: number;
  attempts_count: number;
  last_practiced_at?: string | null;
  created_at: string;
}

export interface LibraryItemCreateInput {
  text: string;
  item_type?: 'word' | 'phrase';
  language: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  category: string;
  focus_phoneme: string;
  meaning_or_tip: string;
}

export interface WeakPhonemeStat {
  phoneme: string;
  error_count: number;
}

export interface PracticeAttemptSummary {
  id: number;
  item_id?: number | null;
  reference_text: string;
  language: string;
  overall_score: number;
  accuracy_score: number;
  fluency_score: number;
  prosody_score: number;
  weak_phonemes: string[];
  created_at: string;
}

export interface LibraryStats {
  total_items: number;
  words_count: number;
  phrases_count: number;
  practiced_count: number;
  mastered_count: number;
  total_attempts: number;
  average_score: number;
  weak_phonemes: WeakPhonemeStat[];
  recent_attempts: PracticeAttemptSummary[];
}

export const LANGUAGE_OPTIONS = [
  { code: 'en', label: 'English (US)', flag: '🇺🇸' },
  { code: 'en-gb', label: 'English (UK)', flag: '🇬🇧' },
  { code: 'es', label: 'Spanish', flag: '🇪🇸' },
  { code: 'fr', label: 'French', flag: '🇫🇷' },
  { code: 'de', label: 'German', flag: '🇩🇪' },
  { code: 'hi', label: 'Hindi', flag: '🇮🇳' },
] as const;

export const CATEGORY_OPTIONS = [
  'Daily Essentials',
  'Tricky Sounds',
  'Work & Interviews',
  'Travel & Dining',
  'Vocabulary',
  'Custom',
] as const;

export const FRIENDLY_PHONEME_GUIDES: Record<
  string,
  { name: string; example: string; howTo: string; mouthShape: string }
> = {
  θ: {
    name: 'Soft "TH" (Unvoiced)',
    example: 'think, thought, three',
    howTo: 'Place the tip of your tongue lightly between your front teeth and blow air gently without buzzing your throat.',
    mouthShape: 'Tongue Between Teeth',
  },
  ð: {
    name: 'Voiced "TH"',
    example: 'the, this, weather',
    howTo: 'Rest your tongue tip lightly between your front teeth and vibrate your vocal cords (feel a gentle buzz in your throat).',
    mouthShape: 'Tongue Between Teeth',
  },
  ʃ: {
    name: '"SH" Sound',
    example: 'she, schedule, pronunciation',
    howTo: 'Pull your tongue slightly back from your teeth, round your lips forward gently, and let air flow smoothly ("shhh").',
    mouthShape: 'Rounded Forward Lips',
  },
  s: {
    name: 'Crisp "S" Sound',
    example: 'see, sea, specific',
    howTo: 'Keep your tongue tip just behind your upper front teeth and smile slightly for a sharp, hiss-like "sss".',
    mouthShape: 'Wide Smile',
  },
  ɹ: {
    name: 'English "R" Sound',
    example: 'right, row, world',
    howTo: 'Curl or bunch your tongue tip slightly back so it floats in the middle of your mouth without touching the roof.',
    mouthShape: 'Slightly Rounded Lips',
  },
  l: {
    name: 'Clear "L" Sound',
    example: 'love, collaborate, world',
    howTo: 'Press the tip of your tongue firmly against the bumpy ridge right behind your upper front teeth.',
    mouthShape: 'Relaxed Open',
  },
  v: {
    name: '"V" Sound',
    example: 'very, have, voice',
    howTo: 'Rest your upper front teeth gently on your bottom lip and buzz your voice.',
    mouthShape: 'Top Teeth on Bottom Lip',
  },
  w: {
    name: '"W" Sound',
    example: 'weather, warm, water',
    howTo: 'Round your lips into a tight small circle ("ooo") and then glide open into the next vowel without touching teeth to lip.',
    mouthShape: 'Tight Circle Lips',
  },
  iː: {
    name: 'Long "EE" Vowel',
    example: 'see, sea, tea',
    howTo: 'Spread your lips into a gentle smile and hold the high "eee" sound slightly longer.',
    mouthShape: 'Wide Smile',
  },
  ɪ: {
    name: 'Short "IH" Vowel',
    example: 'it, think, specific',
    howTo: 'Relax your lips and jaw completely for a quick, short "ih" sound.',
    mouthShape: 'Relaxed Neutral',
  },
  æ: {
    name: 'Open "AA" Vowel',
    example: 'can, cat, have',
    howTo: 'Drop your jaw wider and keep your tongue low and forward in your mouth.',
    mouthShape: 'Wide Open Jaw',
  },
};

export const FALLBACK_SEED_ITEMS: LibraryItem[] = [
  {
    id: 1,
    text: 'Pronunciation',
    item_type: 'word',
    language: 'en',
    difficulty: 'Intermediate',
    category: 'Vocabulary',
    focus_phoneme: 'Stress & /ʃ/',
    meaning_or_tip: "Stress the 4th syllable: pro-nun-ci-AY-shun (notice the 2nd syllable is 'nun', not 'noun').",
    expected_ipa: 'pɹənʌnsiˈeɪʃən',
    words_breakdown: [
      {
        word: 'Pronunciation',
        expected_ipa: 'pɹənʌnsieɪʃən',
        phonemes: ['p', 'ɹ', 'ə', 'n', 'ʌ', 'n', 's', 'i', 'eɪ', 'ʃ', 'ə', 'n'],
        syllables: [
          { syllable_ipa: 'pɹə', stress: 'unstressed' },
          { syllable_ipa: 'nʌn', stress: 'secondary' },
          { syllable_ipa: 'si', stress: 'unstressed' },
          { syllable_ipa: 'eɪ', stress: 'primary' },
          { syllable_ipa: 'ʃən', stress: 'unstressed' },
        ],
      },
    ],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 2,
    text: 'Thought',
    item_type: 'word',
    language: 'en',
    difficulty: 'Beginner',
    category: 'Tricky Sounds',
    focus_phoneme: '/θ/ Unvoiced TH',
    meaning_or_tip: 'Place your tongue tip lightly between your teeth and blow air gently without vibrating your throat.',
    expected_ipa: 'θɔːt',
    words_breakdown: [
      {
        word: 'Thought',
        expected_ipa: 'θɔːt',
        phonemes: ['θ', 'ɔː', 't'],
        syllables: [{ syllable_ipa: 'θɔːt', stress: 'primary' }],
      },
    ],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 3,
    text: 'Weather',
    item_type: 'word',
    language: 'en',
    difficulty: 'Beginner',
    category: 'Daily Essentials',
    focus_phoneme: '/ð/ Voiced TH',
    meaning_or_tip: "Round your lips for 'W', then touch your tongue tip between your front teeth and buzz your throat for 'th'.",
    expected_ipa: 'wɛðɚ',
    words_breakdown: [
      {
        word: 'Weather',
        expected_ipa: 'wɛðɚ',
        phonemes: ['w', 'ɛ', 'ð', 'ɚ'],
        syllables: [
          { syllable_ipa: 'wɛ', stress: 'primary' },
          { syllable_ipa: 'ðɚ', stress: 'unstressed' },
        ],
      },
    ],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 4,
    text: 'Comfortable',
    item_type: 'word',
    language: 'en',
    difficulty: 'Intermediate',
    category: 'Daily Essentials',
    focus_phoneme: 'Syllable Reduction',
    meaning_or_tip: 'Native speakers compress this into 3 or 4 quick beats: KUMF-ter-bul, stressing the first syllable.',
    expected_ipa: 'kʌmfɚtəbəl',
    words_breakdown: [
      {
        word: 'Comfortable',
        expected_ipa: 'kʌmfɚtəbəl',
        phonemes: ['k', 'ʌ', 'm', 'f', 'ɚ', 't', 'ə', 'b', 'əl'],
        syllables: [
          { syllable_ipa: 'kʌmf', stress: 'primary' },
          { syllable_ipa: 'ɚt', stress: 'unstressed' },
          { syllable_ipa: 'əbəl', stress: 'unstressed' },
        ],
      },
    ],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 5,
    text: 'Schedule',
    item_type: 'word',
    language: 'en',
    difficulty: 'Intermediate',
    category: 'Work & Interviews',
    focus_phoneme: '/sk/ + /dʒ/',
    meaning_or_tip: "Start with a crisp 'sk' blend, stress 'SKED', and round your lips slightly for '-jool'.",
    expected_ipa: 'skɛdʒuːl',
    words_breakdown: [
      {
        word: 'Schedule',
        expected_ipa: 'skɛdʒuːl',
        phonemes: ['s', 'k', 'ɛ', 'dʒ', 'uː', 'l'],
        syllables: [
          { syllable_ipa: 'skɛdʒ', stress: 'primary' },
          { syllable_ipa: 'uːl', stress: 'unstressed' },
        ],
      },
    ],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 6,
    text: 'Entrepreneur',
    item_type: 'word',
    language: 'en',
    difficulty: 'Advanced',
    category: 'Work & Interviews',
    focus_phoneme: 'Final Syllable Stress',
    meaning_or_tip: 'Put the strongest, longest beat on the final syllable: ahn-truh-pruh-NUR.',
    expected_ipa: 'ɑːntɹəpɹənɜː',
    words_breakdown: [
      {
        word: 'Entrepreneur',
        expected_ipa: 'ɑːntɹəpɹənɜː',
        phonemes: ['ɑː', 'n', 't', 'ɹ', 'ə', 'p', 'ɹ', 'ə', 'n', 'ɜː'],
        syllables: [
          { syllable_ipa: 'ɑːntɹ', stress: 'secondary' },
          { syllable_ipa: 'əpɹ', stress: 'unstressed' },
          { syllable_ipa: 'ə', stress: 'unstressed' },
          { syllable_ipa: 'nɜː', stress: 'primary' },
        ],
      },
    ],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 7,
    text: 'World',
    item_type: 'word',
    language: 'en',
    difficulty: 'Intermediate',
    category: 'Tricky Sounds',
    focus_phoneme: '/ɜːld/ R-L Cluster',
    meaning_or_tip: "Curl your tongue back for 'er', then glide the tip up behind your top front teeth for 'l' before 'd'.",
    expected_ipa: 'wɜːld',
    words_breakdown: [
      {
        word: 'World',
        expected_ipa: 'wɜːld',
        phonemes: ['w', 'ɜː', 'l', 'd'],
        syllables: [{ syllable_ipa: 'wɜːld', stress: 'primary' }],
      },
    ],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 8,
    text: 'Specific',
    item_type: 'word',
    language: 'en',
    difficulty: 'Intermediate',
    category: 'Work & Interviews',
    focus_phoneme: '2nd-Syllable Stress',
    meaning_or_tip: "Keep the first syllable super short ('spuh') and punch the middle syllable: spuh-SIF-ik.",
    expected_ipa: 'spəsɪfɪk',
    words_breakdown: [
      {
        word: 'Specific',
        expected_ipa: 'spəsɪfɪk',
        phonemes: ['s', 'p', 'ə', 's', 'ɪ', 'f', 'ɪ', 'k'],
        syllables: [
          { syllable_ipa: 'spə', stress: 'unstressed' },
          { syllable_ipa: 'sɪf', stress: 'primary' },
          { syllable_ipa: 'ɪk', stress: 'unstressed' },
        ],
      },
    ],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 9,
    text: 'She can see the sea',
    item_type: 'phrase',
    language: 'en',
    difficulty: 'Beginner',
    category: 'Tricky Sounds',
    focus_phoneme: '/ʃ/ vs /s/',
    meaning_or_tip: "Round your lips slightly for 'She' (/ʃ/), then pull your lip corners back for 'see' and 'sea' (/s/).",
    expected_ipa: 'ʃiː kæn siː ðə siː',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 10,
    text: "I'll think about it today",
    item_type: 'phrase',
    language: 'en',
    difficulty: 'Beginner',
    category: 'Daily Essentials',
    focus_phoneme: 'Connected Speech ‿',
    meaning_or_tip: "Link 'think ‿ about ‿ it' smoothly like one flowing word without pausing between them.",
    expected_ipa: 'aɪl θɪŋk əbaʊt ɪt tədeɪ',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 11,
    text: 'The weather is very warm today',
    item_type: 'phrase',
    language: 'en',
    difficulty: 'Beginner',
    category: 'Daily Essentials',
    focus_phoneme: '/w/ vs /v/',
    meaning_or_tip: "Touch top teeth to bottom lip for 'very' (/v/), and round lips without teeth for 'weather' & 'warm' (/w/).",
    expected_ipa: 'ðə wɛðɚ ɪz vɛɹi wɔːɹm tədeɪ',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 12,
    text: 'Could you help me check this schedule',
    item_type: 'phrase',
    language: 'en',
    difficulty: 'Intermediate',
    category: 'Work & Interviews',
    focus_phoneme: 'Polite Intonation',
    meaning_or_tip: "Blend 'Could ‿ you' smoothly and stress the content words 'help', 'check', and 'schedule'.",
    expected_ipa: 'kʊd juː hɛlp miː tʃɛk ðɪs skɛdʒuːl',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 13,
    text: 'She thought about the pronunciation',
    item_type: 'phrase',
    language: 'en',
    difficulty: 'Intermediate',
    category: 'Tricky Sounds',
    focus_phoneme: '/θ/ & Word Stress',
    meaning_or_tip: "Use a soft unvoiced 'th' on 'thought' and stress the '-AY-' syllable in 'pronunciation'.",
    expected_ipa: 'ʃiː θɔːt əbaʊt ðə pɹənʌnsieɪʃən',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 14,
    text: 'Three free throws in a row',
    item_type: 'phrase',
    language: 'en',
    difficulty: 'Advanced',
    category: 'Tricky Sounds',
    focus_phoneme: '/θɹ/ vs /fɹ/',
    meaning_or_tip: "Switch cleanly between tongue-between-teeth ('three', 'throws') and teeth-on-lip ('free').",
    expected_ipa: 'θɹiː fɹiː θɹoʊz ɪn ə ɹoʊ',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 15,
    text: "We'd love to collaborate on the new project",
    item_type: 'phrase',
    language: 'en',
    difficulty: 'Advanced',
    category: 'Work & Interviews',
    focus_phoneme: 'Business Fluency',
    meaning_or_tip: "Stress 'love', 'la' in collaborate, and 'pro' in project while linking 'collaborate ‿ on'.",
    expected_ipa: 'wiːd lʌv tə kəlæbəɹeɪt ɑːn ðə nuː pɹɑːdʒɛkt',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 16,
    text: 'Water',
    item_type: 'word',
    language: 'en-gb',
    difficulty: 'Beginner',
    category: 'Daily Essentials',
    focus_phoneme: 'Long /ɔː/ Vowel',
    meaning_or_tip: "Round your lips into an oval for a deep British 'WAW-tuh' with a soft non-rhotic ending.",
    expected_ipa: 'wɔːtə',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 17,
    text: 'Can I have a cup of tea please',
    item_type: 'phrase',
    language: 'en-gb',
    difficulty: 'Beginner',
    category: 'Travel & Dining',
    focus_phoneme: 'Weak Forms & Linking',
    meaning_or_tip: "Reduce 'a' and 'of' to quick schwa sounds ('cup ‿ uh tea') and keep 'tea' long.",
    expected_ipa: 'kæn aɪ hæv ə kʌp əv tiː pliːz',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 18,
    text: 'Gracias',
    item_type: 'word',
    language: 'es',
    difficulty: 'Beginner',
    category: 'Daily Essentials',
    focus_phoneme: 'Pure Vowels /a/',
    meaning_or_tip: "Thank you — keep every 'a' vowel crisp, open, and uniform without gliding.",
    expected_ipa: 'ɡɾaθjas',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 19,
    text: 'Buenos días, ¿cómo estás?',
    item_type: 'phrase',
    language: 'es',
    difficulty: 'Beginner',
    category: 'Daily Essentials',
    focus_phoneme: 'Rising Question Pitch',
    meaning_or_tip: "Good morning, how are you? — stress 'dí' and 'tás' with a rising question melody.",
    expected_ipa: 'bwenos dias komo estas',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 20,
    text: 'Bonjour',
    item_type: 'word',
    language: 'fr',
    difficulty: 'Beginner',
    category: 'Daily Essentials',
    focus_phoneme: 'Nasal Vowel & /ʒ/',
    meaning_or_tip: "Hello — soft 'zh' (/ʒ/) in the second syllable with gentle lip rounding.",
    expected_ipa: 'bɔ̃ʒuʁ',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 21,
    text: "Je voudrais un café s'il vous plaît",
    item_type: 'phrase',
    language: 'fr',
    difficulty: 'Beginner',
    category: 'Travel & Dining',
    focus_phoneme: 'Smooth Liaison',
    meaning_or_tip: 'I would like a coffee please — keep syllable timing even and smooth.',
    expected_ipa: 'ʒə vudʁɛ œ̃ kafe sil vu plɛ',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 22,
    text: 'Entschuldigung',
    item_type: 'word',
    language: 'de',
    difficulty: 'Intermediate',
    category: 'Daily Essentials',
    focus_phoneme: '/ʃ/ & Cluster',
    meaning_or_tip: "Excuse me — stress the second syllable ('ent-SHUL-di-gung') with rounded lips for 'sch'.",
    expected_ipa: 'ɛntʃʊldɪɡʊŋ',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 23,
    text: 'Guten Morgen, wie geht es Ihnen',
    item_type: 'phrase',
    language: 'de',
    difficulty: 'Beginner',
    category: 'Daily Essentials',
    focus_phoneme: 'Long Vowels',
    meaning_or_tip: "Good morning, how are you? — hold the 'ee' vowel in 'wie' and 'Ihnen' cleanly.",
    expected_ipa: 'ɡuːtən mɔʁɡən viː ɡeːt ɛs iːnən',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 24,
    text: 'नमस्ते',
    item_type: 'word',
    language: 'hi',
    difficulty: 'Beginner',
    category: 'Daily Essentials',
    focus_phoneme: 'Dental /त/',
    meaning_or_tip: "Hello (Namaste) — touch the tongue tip to the back of your upper teeth for a soft dental 't'.",
    expected_ipa: 'nəməsteː',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
  {
    id: 25,
    text: 'आप कैसे हैं',
    item_type: 'phrase',
    language: 'hi',
    difficulty: 'Beginner',
    category: 'Daily Essentials',
    focus_phoneme: 'Long /आ/ & Diphthong',
    meaning_or_tip: "How are you? (Aap kaise hain) — open your mouth fully for 'आप' and nasalize 'हैं' gently.",
    expected_ipa: 'aːp kɛːseː hɛ̃ː',
    words_breakdown: [],
    best_score: 0,
    attempts_count: 0,
    created_at: new Date().toISOString(),
  },
];

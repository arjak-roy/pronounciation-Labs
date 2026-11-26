import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Award,
  BookOpen,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Edit3,
  Flame,
  Globe,
  Mic,
  Moon,
  Play,
  Plus,
  Search,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Square,
  Sun,
  Target,
  Trash2,
  TrendingUp,
  Upload,
  Volume2,
  X,
} from 'lucide-react';
import {
  ChunkyButton,
  GameCard,
  IconTile,
  Pill,
  ProgressRing,
  Raised,
  StarRow,
  StatTile,
  getGameTone,
  useCardColors,
  useTheme,
} from './game-ui';
import {
  computeClientArticulatoryDiff,
  computeClientConnectedSpeech,
  computeClientWordStress,
  downsampleBuffer,
  encodeMono16kWav,
  launchConfetti,
  mergeBuffers,
  playTone,
  playVictoryChime,
  trimSilenceClientSide,
} from './audio-utils';
import {
  CATEGORY_OPTIONS,
  FALLBACK_SEED_ITEMS,
  FRIENDLY_PHONEME_GUIDES,
  LANGUAGE_OPTIONS,
  type LibraryItem,
  type LibraryItemCreateInput,
  type LibraryStats,
} from './library-data';

declare global {
  interface Window {
    AvatarCoachEngine?: {
      PixiRenderer: new () => any;
      AvatarRuntime: new (opts: { characterUrl: string; renderer: any }) => any;
    };
  }
}

interface PhonemeItem {
  expected: string | null;
  heard: string | null;
  gop_score: number;
  error: 'substitution' | 'deletion' | 'insertion' | null;
  articulatory_diff?: string | null;
}

interface SyllableStressItem {
  syllable_ipa: string;
  stress: 'primary' | 'secondary' | 'unstressed';
  duration_ms?: number;
  score?: number;
}

interface WordStressDiagnostic {
  syllable_count: number;
  expected_primary_index: number;
  detected_peak_index: number;
  stress_correct: boolean;
  stress_pattern_display: string;
  coaching_tip?: string | null;
  tip?: string | null;
  syllables: SyllableStressItem[];
}

interface WordItem {
  word: string;
  score?: number;
  status?: 'correct' | 'warning' | 'mispronounced' | 'omitted';
  expected_ipa: string;
  heard_ipa?: string;
  start_ms?: number;
  end_ms?: number;
  phonemes: (PhonemeItem | string)[];
  syllables?: SyllableStressItem[];
  word_stress?: WordStressDiagnostic | null;
}

interface WordLinkDiagnostic {
  from_word: string;
  to_word: string;
  link_type: 'consonant_to_vowel' | 'consonant_cluster' | 'vowel_to_vowel' | 'normal';
  gap_ms: number;
  status?: 'smooth_link' | 'natural_pause' | 'choppy_pause';
  linked_smoothly?: boolean;
  coaching_tip?: string | null;
  tip?: string | null;
}

interface ConnectedSpeechReport {
  linking_score: number;
  smooth_links_count?: number;
  choppy_Pause_count?: number;
  intonation_contour_hz?: number[];
  intonation_type: string;
  summary_tip: string;
  links: WordLinkDiagnostic[];
}

interface AssessResponse {
  overall_score: number;
  metrics: {
    accuracy: number;
    fluency: number;
    prosody_similarity: number;
  };
  words: WordItem[];
  connected_speech?: ConnectedSpeechReport | null;
}

type MainTab = 'practice' | 'livestt' | 'library' | 'progress';
type ApiTarget = 'aws' | 'local';

const CHARACTERS = [
  { id: 'girl-3d', label: 'Coach Maya', emoji: '👩' },
  { id: 'boy-3d', label: 'Coach Leo', emoji: '👦' },
  { id: 'new-girl', label: 'Coach Aria', emoji: '👩‍🏫' },
  { id: 'kawaii-demo', label: 'Kawaii Coach', emoji: '✨' },
];

function ipaToViseme(ipa: string): string {
  if (!ipa) return 'neutral';
  const s = ipa.toLowerCase();
  if (s.includes('f') || s.includes('v') || s.includes('θ') || s.includes('ð')) return 'fv';
  if (s.includes('ɔ') || s.includes('o') || s.includes('u') || s.includes('ʊ') || s.includes('w')) return 'o';
  if (
    s.includes('i') ||
    s.includes('ɪ') ||
    s.includes('e') ||
    s.includes('s') ||
    s.includes('z') ||
    s.includes('ʃ') ||
    s.includes('ʒ') ||
    s.includes('j')
  ) {
    return 'wide';
  }
  if (
    s.includes('ɑ') ||
    s.includes('æ') ||
    s.includes('ʌ') ||
    s.includes('ɐ') ||
    s.includes('ə') ||
    s.includes('ɜ') ||
    s.includes('a') ||
    s.includes('h')
  ) {
    return 'open';
  }
  return 'neutral';
}

function friendlyVisemeLabel(viseme: string): string {
  switch (viseme) {
    case 'fv':
      return 'Tongue / Lip Touch (TH, F, V)';
    case 'o':
      return 'Rounded Lips (O, OO, W)';
    case 'wide':
      return 'Wide Smile (EE, S, SH)';
    case 'open':
      return 'Open Jaw (AH, AA)';
    case 'smile':
      return 'Relaxed Smile';
    default:
      return 'Neutral Mouth';
  }
}

const LOCAL_LIBRARY_KEY = 'gts_pronunciation_library_v2';
const LOCAL_ATTEMPTS_KEY = 'gts_pronunciation_attempts_v2';

function mergeSeedWithItems(existing: LibraryItem[]): LibraryItem[] {
  const seen = new Set(existing.map((i) => `${i.language}:${i.text.trim().toLowerCase()}`));
  let maxId = existing.reduce((m, i) => Math.max(m, i.id), 0);
  const merged = [...existing];
  for (const seed of FALLBACK_SEED_ITEMS) {
    const key = `${seed.language}:${seed.text.trim().toLowerCase()}`;
    if (!seen.has(key)) {
      maxId += 1;
      merged.push({ ...seed, id: maxId });
      seen.add(key);
    }
  }
  return merged;
}

function loadFallbackLibrary(): LibraryItem[] {
  try {
    const raw = localStorage.getItem(LOCAL_LIBRARY_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) {
        const merged = mergeSeedWithItems(parsed);
        localStorage.setItem(LOCAL_LIBRARY_KEY, JSON.stringify(merged));
        return merged;
      }
    }
  } catch {}
  localStorage.setItem(LOCAL_LIBRARY_KEY, JSON.stringify(FALLBACK_SEED_ITEMS));
  return FALLBACK_SEED_ITEMS;
}

function saveFallbackLibrary(items: LibraryItem[]) {
  try {
    localStorage.setItem(LOCAL_LIBRARY_KEY, JSON.stringify(items));
  } catch {}
}

export default function App() {
  const { isDark, toggleTheme } = useTheme();
  const c = useCardColors();

  // --- Navigation & Settings ---
  const [activeTab, setActiveTab] = useState<MainTab>('practice');
  const [apiTarget, setApiTarget] = useState<ApiTarget>('aws');
  const [serverOnline, setServerOnline] = useState<boolean>(false);
  const [showSettingsModal, setShowSettingsModal] = useState<boolean>(false);
  const [showFilterModal, setShowFilterModal] = useState<boolean>(false);
  const [autoStopVad, setAutoStopVad] = useState<boolean>(true);
  const [soundEffectsOn, setSoundEffectsOn] = useState<boolean>(true);

  const isLocalViteDev =
    typeof window !== 'undefined' &&
    (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') &&
    window.location.port === '5173';

  const apiBase =
    apiTarget === 'aws'
      ? isLocalViteDev
        ? '/aws-api'
        : typeof window !== 'undefined' && window.location.hostname.includes('18-234-54-50')
        ? ''
        : 'https://18-234-54-50.nip.io'
      : isLocalViteDev
      ? '/local-api'
      : 'http://127.0.0.1:8000';

  // --- Library Database State ---
  const [libraryItems, setLibraryItems] = useState<LibraryItem[]>(() => loadFallbackLibrary());
  const [libraryStats, setLibraryStats] = useState<LibraryStats | null>(null);
  const [usingServerDb, setUsingServerDb] = useState<boolean>(false);

  // --- Practice Filters & Active Item (Default: German 'de') ---
  const [selectedLang, setSelectedLang] = useState<string>('de');
  const [filterType, setFilterType] = useState<'all' | 'word' | 'phrase'>('all');
  const [filterDifficulty, setFilterDifficulty] = useState<string>('all');
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const [activeItemId, setActiveItemId] = useState<number>(47); // Default German: "Guten Morgen, wie geht es Ihnen"
  const [referenceText, setReferenceText] = useState<string>('Guten Morgen, wie geht es Ihnen');
  const [speechSpeed, setSpeechSpeed] = useState<number>(1.0);

  // --- Add / Edit Word or Phrase Modal State ---
  const [showItemModal, setShowItemModal] = useState<boolean>(false);
  const [editingItem, setEditingItem] = useState<LibraryItem | null>(null);
  const [formText, setFormText] = useState<string>('');
  const [formType, setFormType] = useState<'word' | 'phrase'>('phrase');
  const [formLang, setFormLang] = useState<string>('de');
  const [formDifficulty, setFormDifficulty] = useState<'Beginner' | 'Intermediate' | 'Advanced'>('Beginner');
  const [formCategory, setFormCategory] = useState<string>('Daily Essentials');
  const [formFocusPhoneme, setFormFocusPhoneme] = useState<string>('');
  const [formTip, setFormTip] = useState<string>('');
  const [savingItem, setSavingItem] = useState<boolean>(false);

  // --- Pronunciation Practice & Assessment State ---
  const [previewWords, setPreviewWords] = useState<WordItem[]>([]);
  const [assessResult, setAssessResult] = useState<AssessResponse | null>(null);
  const [selectedWordIdx, setSelectedWordIdx] = useState<number>(0);
  const [speakingWordIdx, setSpeakingWordIdx] = useState<number | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showResultPopup, setShowResultPopup] = useState<boolean>(false);
  const [countdownValue, setCountdownValue] = useState<number | null>(null);

  // --- Audio Recording & Playback State ---
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isPlayingMyVoice, setIsPlayingMyVoice] = useState<boolean>(false);
  const [liveMicLevel, setLiveMicLevel] = useState<number>(0);
  const [recordedAudioUrl, setRecordedAudioUrl] = useState<string | null>(null);
  const recordedSamplesRef = useRef<Float32Array | null>(null);
  const myVoiceAudioRef = useRef<HTMLAudioElement | null>(null);
  const myVoiceTimerRef = useRef<number | null>(null);

  // --- Mode A: Real-Time Live STT (faster-whisper-small) State ---
  const [isLiveSttStreaming, setIsLiveSttStreaming] = useState<boolean>(false);
  const [isSttRecordingClip, setIsSttRecordingClip] = useState<boolean>(false);
  const [liveSttInterim, setLiveSttInterim] = useState<string>('');
  const [liveSttFinalList, setLiveSttFinalList] = useState<string[]>([]);
  const [liveSttConfidence, setLiveSttConfidence] = useState<number | null>(null);
  const [modeAReady, setModeAReady] = useState<boolean>(false);
  const liveSttWsRef = useRef<WebSocket | null>(null);
  const liveSttChunksRef = useRef<Float32Array[]>([]);
  const isLiveSttRef = useRef<boolean>(false);
  const isSttClipRef = useRef<boolean>(false);

  // --- Learner Gamification State ---
  const [xp, setXp] = useState<number>(() => Number(localStorage.getItem('gts_xp') || '120'));
  const [streak, setStreak] = useState<number>(() => Number(localStorage.getItem('gts_streak') || '2'));

  // --- 3D Avatar Coach State ---
  const [characterId, setCharacterId] = useState<string>('girl-3d');
  const [mouthViseme, setMouthViseme] = useState<string>('neutral');
  const [coachBubble, setCoachBubble] = useState<string>(
    '👋 Tap "Listen to Coach" to hear the phrase, then tap "Speak Now" to practice!'
  );
  const avatarHostRef = useRef<HTMLDivElement | null>(null);
  const avatarRuntimeRef = useRef<any>(null);
  const visemeTimersRef = useRef<number[]>([]);

  // --- Recording Refs ---
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const speechDetectedRef = useRef<boolean>(false);
  const silentFramesRef = useRef<number>(0);
  const isRecordingRef = useRef<boolean>(false);

  // Filtered items for Practice Strip & Library View
  const filteredItems = useMemo(() => {
    return libraryItems.filter((item) => {
      if (selectedLang !== 'all' && item.language !== selectedLang) return false;
      if (filterType !== 'all' && item.item_type !== filterType) return false;
      if (
        filterDifficulty !== 'all' &&
        item.difficulty.toLowerCase() !== filterDifficulty.toLowerCase()
      ) {
        return false;
      }
      if (filterCategory !== 'all' && item.category.toLowerCase() !== filterCategory.toLowerCase()) {
        return false;
      }
      if (searchQuery.trim()) {
        const q = searchQuery.trim().toLowerCase();
        const match =
          item.text.toLowerCase().includes(q) ||
          item.category.toLowerCase().includes(q) ||
          item.focus_phoneme.toLowerCase().includes(q) ||
          item.meaning_or_tip.toLowerCase().includes(q);
        if (!match) return false;
      }
      return true;
    });
  }, [libraryItems, selectedLang, filterType, filterDifficulty, filterCategory, searchQuery]);

  const activeLibraryItem = useMemo(
    () => libraryItems.find((i) => i.id === activeItemId) || null,
    [libraryItems, activeItemId]
  );

  // --- Sound Effects Helper ---
  const triggerTone = (freq: number, dur: number) => {
    if (soundEffectsOn) playTone(freq, dur);
  };

  // --- Load Health, Library Items, and Stats ---
  const fetchHealthAndLibrary = async () => {
    try {
      const hRes = await fetch(`${apiBase}/health`);
      if (hRes.ok) {
        const hData = await hRes.json();
        setServerOnline(Boolean(hData?.mode_b?.ready));
        setModeAReady(Boolean(hData?.mode_a?.ready));
      } else {
        setServerOnline(false);
        setModeAReady(false);
      }
    } catch {
      setServerOnline(false);
    }

    // Try loading items from SQLite backend
    try {
      const iRes = await fetch(`${apiBase}/api/v1/library/items`);
      if (iRes.ok) {
        const items: LibraryItem[] = await iRes.json();
        if (Array.isArray(items) && items.length > 0) {
          const merged = mergeSeedWithItems(items);
          setLibraryItems(merged);
          saveFallbackLibrary(merged);
          setUsingServerDb(true);
        }
      } else {
        setUsingServerDb(false);
      }
    } catch {
      setUsingServerDb(false);
    }

    try {
      const sRes = await fetch(`${apiBase}/api/v1/library/stats`);
      if (sRes.ok) {
        const st: LibraryStats = await sRes.json();
        setLibraryStats(st);
      }
    } catch {}
  };

  useEffect(() => {
    fetchHealthAndLibrary();
  }, [apiBase]);

  // --- Fetch Phonemize Preview when Reference Text or Language Changes ---
  useEffect(() => {
    const timer = window.setTimeout(async () => {
      const text = referenceText.trim();
      if (!text) {
        setPreviewWords([]);
        return;
      }
      try {
        const res = await fetch(
          `${apiBase}/api/v1/pronunciation/phonemize?text=${encodeURIComponent(text)}&language=${encodeURIComponent(selectedLang)}`
        );
        if (res.ok) {
          const data = await res.json();
          setPreviewWords(data.words || []);
          return;
        }
      } catch {}
      setPreviewWords(
        text.split(/\s+/).map((w) => ({
          word: w,
          expected_ipa: '...',
          phonemes: [],
        }))
      );
    }, 180);
    return () => window.clearTimeout(timer);
  }, [referenceText, selectedLang, apiBase]);

  // --- Initialize 3D Avatar Coach ---
  useEffect(() => {
    let cancelled = false;
    async function bootAvatar() {
      if (!window.AvatarCoachEngine || !avatarHostRef.current) return;
      try {
        if (avatarRuntimeRef.current) {
          avatarRuntimeRef.current.destroy();
          avatarRuntimeRef.current = null;
        }
        avatarHostRef.current.innerHTML = '';
        const renderer = new window.AvatarCoachEngine.PixiRenderer();
        const rt = new window.AvatarCoachEngine.AvatarRuntime({
          characterUrl: `/characters/${characterId}`,
          renderer,
        });
        await rt.ready;
        if (cancelled || !avatarHostRef.current) return;
        const canvas = renderer.getSurfaceElement();
        if (canvas) {
          canvas.style.width = '100%';
          canvas.style.height = '100%';
          canvas.style.objectFit = 'contain';
          avatarHostRef.current.appendChild(canvas);
        }
        await rt.load(`/characters/${characterId}`);
        rt.on('mouth.changed', ({ mouth }: { mouth: string }) => {
          setMouthViseme(mouth);
        });
        avatarRuntimeRef.current = rt;
      } catch (e) {
        console.warn('Avatar Coach init notice:', e);
      }
    }
    bootAvatar();
    return () => {
      cancelled = true;
    };
  }, [characterId]);

  const clearVisemeTimers = () => {
    visemeTimersRef.current.forEach((id) => window.clearTimeout(id));
    visemeTimersRef.current = [];
  };

  const setAvatarViseme = (viseme: string) => {
    setMouthViseme(viseme);
    if (avatarRuntimeRef.current) {
      avatarRuntimeRef.current.setMouth(viseme);
    }
  };

  const animateWordVisemes = (wordObj: WordItem | undefined, durationMs = 380) => {
    if (!wordObj) return;
    clearVisemeTimers();
    const list = (wordObj.phonemes || [])
      .map((p) => (typeof p === 'string' ? p : p.expected || p.heard || ''))
      .filter(Boolean);
    if (list.length === 0) {
      setAvatarViseme(ipaToViseme(wordObj.expected_ipa));
      return;
    }
    const step = Math.max(65, durationMs / list.length);
    list.forEach((ph, i) => {
      const tid = window.setTimeout(() => {
        setAvatarViseme(ipaToViseme(ph));
      }, i * step);
      visemeTimersRef.current.push(tid);
    });
    visemeTimersRef.current.push(
      window.setTimeout(() => setAvatarViseme('neutral'), list.length * step + 70)
    );
  };

  // --- Select a Word or Phrase from the Library ---
  const selectLibraryItemForPractice = (item: LibraryItem) => {
    setActiveItemId(item.id);
    setSelectedLang(item.language);
    setReferenceText(item.text);
    setAssessResult(null);
    setSelectedWordIdx(0);
    setShowResultPopup(false);
    setErrorMsg(null);
    setActiveTab('practice');
    setCoachBubble(
      item.meaning_or_tip
        ? `💡 <strong>Coach Tip:</strong> ${item.meaning_or_tip}`
        : `👋 Ready to practice "<strong>${item.text}</strong>"! Listen first or speak whenever you're ready.`
    );
    triggerTone(520, 0.05);
  };

  const stepDeckItem = (delta: number) => {
    if (filteredItems.length === 0) return;
    const curIdx = filteredItems.findIndex((i) => i.id === activeItemId);
    const nextIdx = curIdx === -1 ? 0 : (curIdx + delta + filteredItems.length) % filteredItems.length;
    selectLibraryItemForPractice(filteredItems[nextIdx]);
  };

  // --- Create / Update / Delete Word or Phrase in Database ---
  const openCreateModal = (defaultType: 'word' | 'phrase' = 'phrase') => {
    setEditingItem(null);
    setFormText('');
    setFormType(defaultType);
    setFormLang(selectedLang === 'all' ? 'en' : selectedLang);
    setFormDifficulty('Beginner');
    setFormCategory('Daily Essentials');
    setFormFocusPhoneme('');
    setFormTip('');
    setShowItemModal(true);
  };

  const openEditModal = (item: LibraryItem) => {
    setEditingItem(item);
    setFormText(item.text);
    setFormType(item.item_type);
    setFormLang(item.language);
    setFormDifficulty(
      (item.difficulty as 'Beginner' | 'Intermediate' | 'Advanced') || 'Beginner'
    );
    setFormCategory(item.category || 'Daily Essentials');
    setFormFocusPhoneme(item.focus_phoneme || '');
    setFormTip(item.meaning_or_tip || '');
    setShowItemModal(true);
  };

  const handleSaveLibraryItem = async (e: React.FormEvent) => {
    e.preventDefault();
    const cleanText = formText.trim();
    if (!cleanText) return;
    setSavingItem(true);

    const inferredType: 'word' | 'phrase' =
      cleanText.split(/\s+/).length === 1 ? 'word' : formType;

    const payload: LibraryItemCreateInput = {
      text: cleanText,
      item_type: inferredType,
      language: formLang,
      difficulty: formDifficulty,
      category: formCategory,
      focus_phoneme: formFocusPhoneme.trim(),
      meaning_or_tip: formTip.trim(),
    };

    try {
      if (editingItem) {
        const res = await fetch(`${apiBase}/api/v1/library/items/${editingItem.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (res.ok) {
          const updated: LibraryItem = await res.json();
          const next = libraryItems.map((it) => (it.id === updated.id ? updated : it));
          setLibraryItems(next);
          saveFallbackLibrary(next);
          setShowItemModal(false);
          setSavingItem(false);
          return;
        }
      } else {
        const res = await fetch(`${apiBase}/api/v1/library/items`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (res.ok) {
          const created: LibraryItem = await res.json();
          const next = [...libraryItems, created];
          setLibraryItems(next);
          saveFallbackLibrary(next);
          setShowItemModal(false);
          setSavingItem(false);
          selectLibraryItemForPractice(created);
          return;
        }
      }
    } catch {}

    // Client-side fallback if remote container hasn't restarted with /api/v1/library yet
    let ipa = '';
    let wordsBreakdown: any[] = [];
    try {
      const pRes = await fetch(
        `${apiBase}/api/v1/pronunciation/phonemize?text=${encodeURIComponent(cleanText)}&language=${encodeURIComponent(formLang)}`
      );
      if (pRes.ok) {
        const pData = await pRes.json();
        wordsBreakdown = pData.words || [];
        ipa = wordsBreakdown.map((w: any) => w.expected_ipa).join(' ');
      }
    } catch {}

    if (editingItem) {
      const updated: LibraryItem = {
        ...editingItem,
        text: cleanText,
        item_type: inferredType,
        language: formLang,
        difficulty: formDifficulty,
        category: formCategory,
        focus_phoneme: formFocusPhoneme.trim(),
        meaning_or_tip: formTip.trim(),
        expected_ipa: ipa || editingItem.expected_ipa,
        words_breakdown: wordsBreakdown.length > 0 ? wordsBreakdown : editingItem.words_breakdown,
      };
      const next = libraryItems.map((it) => (it.id === updated.id ? updated : it));
      setLibraryItems(next);
      saveFallbackLibrary(next);
    } else {
      const nextId = libraryItems.reduce((max, it) => Math.max(max, it.id), 0) + 1;
      const created: LibraryItem = {
        id: nextId,
        text: cleanText,
        item_type: inferredType,
        language: formLang,
        difficulty: formDifficulty,
        category: formCategory,
        focus_phoneme: formFocusPhoneme.trim(),
        meaning_or_tip: formTip.trim(),
        expected_ipa: ipa,
        words_breakdown: wordsBreakdown,
        best_score: 0,
        attempts_count: 0,
        created_at: new Date().toISOString(),
      };
      const next = [...libraryItems, created];
      setLibraryItems(next);
      saveFallbackLibrary(next);
      selectLibraryItemForPractice(created);
    }

    setShowItemModal(false);
    setSavingItem(false);
  };

  const handleDeleteLibraryItem = async (id: number) => {
    try {
      await fetch(`${apiBase}/api/v1/library/items/${id}`, { method: 'DELETE' });
    } catch {}
    const next = libraryItems.filter((it) => it.id !== id);
    setLibraryItems(next);
    saveFallbackLibrary(next);
  };

  // --- Speak Phrase or Word via Coach TTS ---
  const speakCoachPhrase = (customText?: string, onFinish?: () => void) => {
    const textToSpeak = (customText || referenceText).trim();
    if (!textToSpeak) return;
    window.speechSynthesis.cancel();

    const u = new SpeechSynthesisUtterance(textToSpeak);
    u.lang = selectedLang === 'all' ? 'en' : selectedLang;
    u.rate = speechSpeed;

    const words = assessResult?.words || previewWords;
    const approxMs = Math.max(900, (words.length * 360) / speechSpeed);
    const stepMs = approxMs / Math.max(1, words.length);
    const timers: number[] = [];

    u.onstart = () => {
      if (avatarRuntimeRef.current) {
        avatarRuntimeRef.current.setActivity('speaking');
        avatarRuntimeRef.current.setExpression('happy', 0.65);
      }
      setCoachBubble(`🔊 Listen closely to my rhythm: "<strong>${textToSpeak}</strong>"`);
      words.forEach((w, idx) => {
        const tid = window.setTimeout(() => {
          setSpeakingWordIdx(idx);
          animateWordVisemes(w, stepMs * 0.85);
        }, idx * stepMs);
        timers.push(tid);
      });
    };

    u.onend = () => {
      timers.forEach((t) => window.clearTimeout(t));
      clearVisemeTimers();
      setSpeakingWordIdx(null);
      if (avatarRuntimeRef.current) {
        avatarRuntimeRef.current.setActivity('idle');
        setAvatarViseme('neutral');
      }
      setCoachBubble('🎙️ Your turn! Tap "Speak Now" and repeat it naturally.');
      if (onFinish) onFinish();
    };

    window.speechSynthesis.speak(u);
  };

  // --- Guided 3-2-1 Countdown Practice ---
  const startGuidedPractice = () => {
    if (isRecording) return;
    speakCoachPhrase(undefined, () => {
      let count = 3;
      setCountdownValue(count);
      triggerTone(440, 0.08);
      const iv = window.setInterval(() => {
        count -= 1;
        if (count > 0) {
          setCountdownValue(count);
          triggerTone(440, 0.08);
        } else {
          window.clearInterval(iv);
          setCountdownValue(null);
          triggerTone(880, 0.14);
          startRecording();
        }
      }, 650);
    });
  };

  // --- Play My Voice (Karaoke Word Sync + Slice Playback) ---
  const playMyVoice = () => {
    if (!recordedAudioUrl) return;
    if (myVoiceAudioRef.current) {
      myVoiceAudioRef.current.pause();
      myVoiceAudioRef.current = null;
    }
    if (myVoiceTimerRef.current) {
      window.clearInterval(myVoiceTimerRef.current);
      myVoiceTimerRef.current = null;
    }

    const audio = new Audio(recordedAudioUrl);
    myVoiceAudioRef.current = audio;
    setIsPlayingMyVoice(true);
    const words = assessResult?.words || [];

    audio.onplay = () => {
      if (avatarRuntimeRef.current) {
        avatarRuntimeRef.current.setActivity('listening');
      }
      setCoachBubble('🎧 Replaying your recording — watch each word light up!');
      myVoiceTimerRef.current = window.setInterval(() => {
        const curMs = audio.currentTime * 1000;
        const activeIdx = words.findIndex(
          (w) =>
            typeof w.start_ms === 'number' &&
            typeof w.end_ms === 'number' &&
            curMs >= w.start_ms &&
            curMs <= w.end_ms + 70
        );
        if (activeIdx !== -1) {
          setSpeakingWordIdx(activeIdx);
          animateWordVisemes(words[activeIdx], 180);
        }
      }, 35);
    };

    const cleanup = () => {
      if (myVoiceTimerRef.current) {
        window.clearInterval(myVoiceTimerRef.current);
        myVoiceTimerRef.current = null;
      }
      setSpeakingWordIdx(null);
      setIsPlayingMyVoice(false);
      setAvatarViseme('neutral');
    };

    audio.onended = cleanup;
    audio.onpause = cleanup;
    audio.play();
  };

  const playSelectedWordSlice = (wordObj: WordItem, idx: number) => {
    const samples = recordedSamplesRef.current;
    if (
      !samples ||
      typeof wordObj.start_ms !== 'number' ||
      typeof wordObj.end_ms !== 'number'
    ) {
      playMyVoice();
      return;
    }
    const startSample = Math.max(0, Math.floor((wordObj.start_ms / 1000) * 16000) - 480);
    const endSample = Math.min(
      samples.length,
      Math.ceil((wordObj.end_ms / 1000) * 16000) + 640
    );
    const slice = samples.slice(startSample, Math.max(startSample + 320, endSample));
    const blob = encodeMono16kWav(slice, 16000);
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    setSpeakingWordIdx(idx);
    animateWordVisemes(wordObj, Math.max(260, wordObj.end_ms - wordObj.start_ms));
    audio.onended = () => {
      setSpeakingWordIdx(null);
      URL.revokeObjectURL(url);
    };
    audio.play();
  };

  // --- Start & Stop Microphone Recording ---
  const startRecording = async () => {
    try {
      setErrorMsg(null);
      setShowResultPopup(false);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const inRate = ctx.sampleRate;
      chunksRef.current = [];
      speechDetectedRef.current = false;
      silentFramesRef.current = 0;

      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(2048, 1, 1);

      processor.onaudioprocess = (e) => {
        if (!isRecordingRef.current) return;
        const ch = e.inputBuffer.getChannelData(0);
        chunksRef.current.push(new Float32Array(ch));

        let sum = 0;
        for (let i = 0; i < ch.length; i++) sum += ch[i] * ch[i];
        const rms = Math.sqrt(sum / ch.length);
        setLiveMicLevel(Math.min(100, Math.round(rms * 420)));

        if (rms > 0.024) {
          speechDetectedRef.current = true;
          silentFramesRef.current = 0;
        } else if (speechDetectedRef.current) {
          silentFramesRef.current += 1;
          const framesFor950ms = Math.ceil((0.95 * inRate) / 2048);
          if (autoStopVad && silentFramesRef.current >= framesFor950ms) {
            stopRecordingAndScore();
          }
        }
      };

      source.connect(processor);
      processor.connect(ctx.destination);

      mediaStreamRef.current = stream;
      audioCtxRef.current = ctx;
      processorRef.current = processor;
      isRecordingRef.current = true;
      setIsRecording(true);
      triggerTone(660, 0.07);

      if (avatarRuntimeRef.current) {
        avatarRuntimeRef.current.setActivity('listening');
        avatarRuntimeRef.current.setHeadTurn(0.1);
      }
      setCoachBubble('👂 Listening... Speak clearly, and I will score automatically when you finish!');
    } catch (err: any) {
      setErrorMsg(`Microphone access needed: ${err?.message || 'Please allow microphone access'}`);
    }
  };

  const stopRecordingAndScore = async () => {
    if (!isRecordingRef.current) return;
    isRecordingRef.current = false;
    setIsRecording(false);
    setLiveMicLevel(0);
    triggerTone(440, 0.07);

    const inRate = audioCtxRef.current?.sampleRate || 48000;
    if (processorRef.current) processorRef.current.disconnect();
    if (mediaStreamRef.current) mediaStreamRef.current.getTracks().forEach((t) => t.stop());
    if (audioCtxRef.current) {
      try {
        await audioCtxRef.current.close();
      } catch {}
    }

    const merged = mergeBuffers(chunksRef.current);
    const ds16k = downsampleBuffer(merged, inRate, 16000);
    const trimmed = trimSilenceClientSide(ds16k, 16000);
    recordedSamplesRef.current = trimmed;
    const wavBlob = encodeMono16kWav(trimmed, 16000);
    if (recordedAudioUrl) URL.revokeObjectURL(recordedAudioUrl);
    setRecordedAudioUrl(URL.createObjectURL(wavBlob));
    await submitAudioForAssessment(wavBlob, 'recording_16k.wav');
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';
    const { wavBlob } = await normalizeUploadedAudioFile(file);
    if (recordedAudioUrl) URL.revokeObjectURL(recordedAudioUrl);
    setRecordedAudioUrl(URL.createObjectURL(wavBlob));
    await submitAudioForAssessment(wavBlob, 'uploaded_16k.wav');
  };

  // Decode any browser-supported audio file (.wav, .mp3, .m4a, .webm, .ogg) into 16kHz mono WAV
  const normalizeUploadedAudioFile = async (
    file: File
  ): Promise<{ wavBlob: Blob; samples16k: Float32Array | null }> => {
    try {
      const arrayBuf = await file.arrayBuffer();
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const decoded = await ctx.decodeAudioData(arrayBuf);
      const raw = decoded.getChannelData(0);
      const ds16k = downsampleBuffer(raw, decoded.sampleRate, 16000);
      const trimmed = trimSilenceClientSide(ds16k, 16000);
      await ctx.close();
      recordedSamplesRef.current = trimmed;
      return { wavBlob: encodeMono16kWav(trimmed, 16000), samples16k: trimmed };
    } catch {
      return { wavBlob: file, samples16k: null };
    }
  };

  // Shared helper to transcribe a 16kHz WAV blob (either Transcribe Only or Transcribe + Score)
  const transcribeOrScoreWavBlob = async (
    wavBlob: Blob,
    filename: string,
    alsoScorePronunciation: boolean
  ) => {
    setIsAnalyzing(true);
    setErrorMsg(null);
    const t0 = performance.now();

    try {
      if (recordedAudioUrl) URL.revokeObjectURL(recordedAudioUrl);
      setRecordedAudioUrl(URL.createObjectURL(wavBlob));

      const cleanLang = selectedLang === 'all' ? 'de' : selectedLang;
      const fd = new FormData();
      fd.append('audio', wavBlob, filename);
      fd.append('language', cleanLang);

      if (alsoScorePronunciation) {
        const res = await fetch(`${apiBase}/api/v1/pronunciation/free-assess`, {
          method: 'POST',
          body: fd,
        });
        const elapsed = Math.round(performance.now() - t0);
        setLatencyMs(elapsed);

        const minWait = Math.max(0, 700 - (performance.now() - t0));
        if (minWait > 0) await new Promise((r) => window.setTimeout(r, minWait));

        const data = await res.json();
        if (!res.ok) {
          setErrorMsg(data?.detail || 'Could not transcribe audio.');
          return;
        }
        if (data.detected_transcript) {
          setLiveSttFinalList([data.detected_transcript]);
          setLiveSttInterim('');
          if (typeof data.stt_confidence === 'number') {
            setLiveSttConfidence(data.stt_confidence);
          }
          setReferenceText(data.detected_transcript);
          setAssessResult({
            overall_score: data.overall_score,
            metrics: data.metrics,
            words: data.words,
            connected_speech: data.connected_speech || computeClientConnectedSpeech(data.words),
          });
          setShowResultPopup(true);
        }
      } else {
        let res = await fetch(`${apiBase}/api/v1/stt/transcribe`, {
          method: 'POST',
          body: fd,
        });
        if (res.status === 404) {
          const fd2 = new FormData();
          fd2.append('audio', wavBlob, filename);
          fd2.append('language', cleanLang);
          res = await fetch(`${apiBase}/api/v1/pronunciation/free-assess`, {
            method: 'POST',
            body: fd2,
          });
        }
        const elapsed = Math.round(performance.now() - t0);
        setLatencyMs(elapsed);

        const data = await res.json();
        if (!res.ok) {
          setErrorMsg(data?.detail || 'Could not transcribe audio.');
          return;
        }
        const text = data.transcript || data.detected_transcript || '';
        const conf =
          typeof data.confidence === 'number'
            ? data.confidence
            : typeof data.stt_confidence === 'number'
            ? data.stt_confidence
            : null;
        if (text) {
          setLiveSttFinalList([text]);
          setLiveSttInterim('');
          if (conf !== null) setLiveSttConfidence(conf);
        } else {
          setErrorMsg('No clear speech detected in the audio.');
        }
      }
    } catch (err: any) {
      setErrorMsg(`Transcription error: ${err?.message || 'Failed to process audio'}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Upload Audio in Live STT Tab (either Transcribe Only or Transcribe + Score Pronunciation)
  const handleSttFileUpload = async (
    e: React.ChangeEvent<HTMLInputElement>,
    alsoScorePronunciation: boolean
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';
    const { wavBlob } = await normalizeUploadedAudioFile(file);
    await transcribeOrScoreWavBlob(wavBlob, 'uploaded_stt_16k.wav', alsoScorePronunciation);
  };

  // Record a Voice Clip in Live STT Tab to Transcribe (or Transcribe & Score)
  const startSttClipRecording = async () => {
    try {
      setErrorMsg(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      liveSttChunksRef.current = [];

      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(2048, 1, 1);

      processor.onaudioprocess = (e) => {
        if (!isSttClipRef.current) return;
        const ch = new Float32Array(e.inputBuffer.getChannelData(0));
        liveSttChunksRef.current.push(ch);

        let sum = 0;
        for (let i = 0; i < ch.length; i++) sum += ch[i] * ch[i];
        const rms = Math.sqrt(sum / ch.length);
        setLiveMicLevel(Math.min(100, Math.round(rms * 420)));
      };

      source.connect(processor);
      processor.connect(ctx.destination);

      mediaStreamRef.current = stream;
      audioCtxRef.current = ctx;
      processorRef.current = processor;
      isSttClipRef.current = true;
      setIsSttRecordingClip(true);
      triggerTone(660, 0.07);
    } catch (err: any) {
      setErrorMsg(`Microphone error: ${err?.message || 'Please allow microphone access'}`);
    }
  };

  const stopSttClipRecording = async (alsoScorePronunciation: boolean) => {
    if (!isSttClipRef.current) return;
    isSttClipRef.current = false;
    setIsSttRecordingClip(false);
    setLiveMicLevel(0);
    triggerTone(440, 0.07);

    const inRate = audioCtxRef.current?.sampleRate || 48000;
    if (processorRef.current) processorRef.current.disconnect();
    if (mediaStreamRef.current) mediaStreamRef.current.getTracks().forEach((t) => t.stop());
    if (audioCtxRef.current) {
      try {
        await audioCtxRef.current.close();
      } catch {}
    }

    const merged = mergeBuffers(liveSttChunksRef.current);
    if (merged.length < 1600) {
      setErrorMsg('Recording was too short. Please speak for at least 1 second.');
      return;
    }
    const ds16k = downsampleBuffer(merged, inRate, 16000);
    const trimmed = trimSilenceClientSide(ds16k, 16000);
    recordedSamplesRef.current = trimmed;
    const wavBlob = encodeMono16kWav(trimmed, 16000);
    await transcribeOrScoreWavBlob(wavBlob, 'recorded_stt_16k.wav', alsoScorePronunciation);
  };

  // --- Mode A: Real-Time Live STT Streaming (faster-whisper-small) + Free-Speech Scoring ---
  const buildWsUrl = (lang: string): string => {
    const cleanLang = lang === 'all' ? 'de' : lang;
    if (typeof window === 'undefined') return '';
    if (apiBase.startsWith('http://') || apiBase.startsWith('https://')) {
      const wsBase = apiBase.replace(/^http/, 'ws');
      return `${wsBase}/ws/transcribe?lang=${encodeURIComponent(cleanLang)}`;
    }
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.host}${apiBase}/ws/transcribe?lang=${encodeURIComponent(cleanLang)}`;
  };

  const startLiveSttStream = async () => {
    try {
      setErrorMsg(null);
      setLiveSttInterim('');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const inRate = ctx.sampleRate;
      liveSttChunksRef.current = [];

      const wsUrl = buildWsUrl(selectedLang);
      const ws = new WebSocket(wsUrl);
      liveSttWsRef.current = ws;

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.error) {
            setErrorMsg(msg.error);
            return;
          }
          if (msg.type === 'interim') {
            setLiveSttInterim(msg.transcript || '');
            if (typeof msg.confidence === 'number') setLiveSttConfidence(msg.confidence);
          } else if (msg.type === 'final' && msg.transcript) {
            setLiveSttFinalList((prev) => [...prev, msg.transcript]);
            setLiveSttInterim('');
            if (typeof msg.confidence === 'number') setLiveSttConfidence(msg.confidence);
          }
        } catch {}
      };

      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(2048, 1, 1);
      let accChunks: Float32Array[] = [];
      let accSamples = 0;

      processor.onaudioprocess = (e) => {
        if (!isLiveSttRef.current) return;
        const ch = new Float32Array(e.inputBuffer.getChannelData(0));
        liveSttChunksRef.current.push(ch);

        let sum = 0;
        for (let i = 0; i < ch.length; i++) sum += ch[i] * ch[i];
        const rms = Math.sqrt(sum / ch.length);
        setLiveMicLevel(Math.min(100, Math.round(rms * 420)));

        const ds16k = downsampleBuffer(ch, inRate, 16000);
        accChunks.push(ds16k);
        accSamples += ds16k.length;

        if (accSamples >= 2400 && ws.readyState === WebSocket.OPEN) {
          const merged16k = mergeBuffers(accChunks);
          accChunks = [];
          accSamples = 0;
          const pcm16 = new Int16Array(merged16k.length);
          for (let i = 0; i < merged16k.length; i++) {
            const s = Math.max(-1, Math.min(1, merged16k[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          ws.send(pcm16.buffer);
        }
      };

      source.connect(processor);
      processor.connect(ctx.destination);

      mediaStreamRef.current = stream;
      audioCtxRef.current = ctx;
      processorRef.current = processor;
      isLiveSttRef.current = true;
      setIsLiveSttStreaming(true);
      triggerTone(660, 0.07);
    } catch (err: any) {
      setErrorMsg(`Microphone error: ${err?.message || 'Please allow microphone access'}`);
    }
  };

  const stopLiveSttStream = async (andScorePronunciation = false) => {
    if (!isLiveSttRef.current) return;
    isLiveSttRef.current = false;
    setIsLiveSttStreaming(false);
    setLiveMicLevel(0);
    triggerTone(440, 0.07);

    const inRate = audioCtxRef.current?.sampleRate || 48000;
    if (processorRef.current) processorRef.current.disconnect();
    if (mediaStreamRef.current) mediaStreamRef.current.getTracks().forEach((t) => t.stop());
    if (audioCtxRef.current) {
      try {
        await audioCtxRef.current.close();
      } catch {}
    }

    if (liveSttWsRef.current && liveSttWsRef.current.readyState === WebSocket.OPEN) {
      try {
        liveSttWsRef.current.send(JSON.stringify({ type: 'close' }));
      } catch {}
    }

    const merged = mergeBuffers(liveSttChunksRef.current);
    if (merged.length < 1600) return;
    const ds16k = downsampleBuffer(merged, inRate, 16000);
    const trimmed = trimSilenceClientSide(ds16k, 16000);
    recordedSamplesRef.current = trimmed;
    const wavBlob = encodeMono16kWav(trimmed, 16000);
    if (recordedAudioUrl) URL.revokeObjectURL(recordedAudioUrl);
    setRecordedAudioUrl(URL.createObjectURL(wavBlob));

    if (andScorePronunciation) {
      setIsAnalyzing(true);
      setErrorMsg(null);
      const t0 = performance.now();
      try {
        const cleanLang = selectedLang === 'all' ? 'de' : selectedLang;
        const fd = new FormData();
        fd.append('audio', wavBlob, 'free_speech_16k.wav');
        fd.append('language', cleanLang);

        const res = await fetch(`${apiBase}/api/v1/pronunciation/free-assess`, {
          method: 'POST',
          body: fd,
        });
        const elapsed = Math.round(performance.now() - t0);
        setLatencyMs(elapsed);

        const minWait = Math.max(0, 750 - (performance.now() - t0));
        if (minWait > 0) {
          await new Promise((r) => window.setTimeout(r, minWait));
        }

        const data = await res.json();
        if (res.ok && data.detected_transcript) {
          setLiveSttFinalList([data.detected_transcript]);
          setLiveSttInterim('');
          if (typeof data.stt_confidence === 'number') {
            setLiveSttConfidence(data.stt_confidence);
          }
          setReferenceText(data.detected_transcript);
          setAssessResult({
            overall_score: data.overall_score,
            metrics: data.metrics,
            words: data.words,
            connected_speech: data.connected_speech || computeClientConnectedSpeech(data.words),
          });
          setShowResultPopup(true);
        } else {
          const fallbackText = [...liveSttFinalList, liveSttInterim].join(' ').trim();
          if (fallbackText) {
            setReferenceText(fallbackText);
            await submitAudioForAssessment(wavBlob, 'free_speech_16k.wav');
          } else {
            setErrorMsg(data?.detail || 'No clear speech detected. Please speak slightly louder.');
          }
        }
      } catch (err: any) {
        setErrorMsg(`Free-speech scoring error: ${err?.message || 'Unable to score audio'}`);
      } finally {
        setIsAnalyzing(false);
      }
    }
  };

  // --- Submit Audio to Mode B Pronunciation Engine ---
  const submitAudioForAssessment = async (blob: Blob, filename: string) => {
    const cleanRef = referenceText.trim();
    if (!cleanRef) return;

    setIsAnalyzing(true);
    setErrorMsg(null);
    if (avatarRuntimeRef.current) {
      avatarRuntimeRef.current.setActivity('thinking');
      avatarRuntimeRef.current.setExpression('surprised', 0.45);
      if (typeof avatarRuntimeRef.current.setGaze === 'function') {
        avatarRuntimeRef.current.setGaze('up');
      }
      avatarRuntimeRef.current.setHeadTurn(0.08);
    }
    setCoachBubble('🤔 Hmm, let me listen closely to your sounds, syllable stress, and flow...');

    const t0 = performance.now();
    try {
      const fd = new FormData();
      fd.append('audio', blob, filename);
      fd.append('reference_text', cleanRef);
      fd.append('language', selectedLang === 'all' ? 'en' : selectedLang);
      if (activeLibraryItem) {
        fd.append('sentence_id', String(activeLibraryItem.id));
      }

      const res = await fetch(`${apiBase}/api/v1/pronunciation/assess`, {
        method: 'POST',
        body: fd,
      });
      const elapsed = Math.round(performance.now() - t0);
      setLatencyMs(elapsed);

      // Keep the thinking avatar screen visible for at least 750ms so the transition feels natural
      const minWait = Math.max(0, 750 - (performance.now() - t0));
      if (minWait > 0) {
        await new Promise((resolve) => window.setTimeout(resolve, minWait));
      }

      const data = await res.json();
      if (!res.ok) {
        setErrorMsg(data?.detail || 'Could not score recording. Please try speaking slightly louder.');
        return;
      }

      // Enrich with client-side fallbacks if needed
      const enrichedWords: WordItem[] = (data.words || []).map((w: WordItem) => {
        const phonemes = (w.phonemes || []).map((p) => {
          if (typeof p === 'string') return p;
          if (p.error && !p.articulatory_diff) {
            return {
              ...p,
              articulatory_diff: computeClientArticulatoryDiff(p.expected, p.heard),
            };
          }
          return p;
        });
        const word_stress =
          w.word_stress ||
          computeClientWordStress(w.expected_ipa, phonemes as any, w.start_ms, w.end_ms);
        return { ...w, phonemes, word_stress };
      });

      const connected_speech =
        data.connected_speech || computeClientConnectedSpeech(enrichedWords as any);

      const finalResult: AssessResponse = {
        ...data,
        words: enrichedWords,
        connected_speech,
      };

      setAssessResult(finalResult);
      setShowResultPopup(true);

      // Update local library item best_score & attempts_count
      const score = finalResult.overall_score;
      const updatedItems = libraryItems.map((item) => {
        if (
          item.id === activeItemId ||
          item.text.toLowerCase() === cleanRef.toLowerCase()
        ) {
          return {
            ...item,
            best_score: Math.max(item.best_score || 0, score),
            attempts_count: (item.attempts_count || 0) + 1,
            last_practiced_at: new Date().toISOString(),
          };
        }
        return item;
      });
      setLibraryItems(updatedItems);
      saveFallbackLibrary(updatedItems);

      // Refresh server stats if available
      fetch(`${apiBase}/api/v1/library/stats`)
        .then((r) => (r.ok ? r.json() : null))
        .then((st) => {
          if (st) setLibraryStats(st);
        })
        .catch(() => {});

      // Gamification rewards
      const stars = score >= 90 ? 3 : score >= 75 ? 2 : score >= 55 ? 1 : 0;
      const nextStreak = score >= 80 ? streak + 1 : 0;
      const gainedXp = 15 + stars * 10;
      setStreak(nextStreak);
      setXp((prev) => {
        const v = prev + gainedXp;
        localStorage.setItem('gts_xp', String(v));
        return v;
      });
      localStorage.setItem('gts_streak', String(nextStreak));

      if (stars >= 2) {
        if (soundEffectsOn) playVictoryChime(stars === 3);
        if (stars === 3) launchConfetti();
        if (avatarRuntimeRef.current) {
          avatarRuntimeRef.current.setActivity('idle');
          avatarRuntimeRef.current.setExpression('happy', 1.0);
          avatarRuntimeRef.current.playGesture('nod');
          setAvatarViseme('smile');
        }
        setCoachBubble(
          `🎉 Wonderful job! You scored <strong>${score}%</strong>! Tap any word below to inspect your sounds.`
        );
      } else {
        triggerTone(320, 0.14);
        if (avatarRuntimeRef.current) {
          avatarRuntimeRef.current.setActivity('idle');
          avatarRuntimeRef.current.setExpression('concerned', 0.85);
        }
        setCoachBubble(
          `💡 You scored <strong>${score}%</strong>. Let's look at the highlighted word below to see how to position your tongue & lips!`
        );
      }

      // Select lowest scoring word automatically for helpful coaching
      if (enrichedWords.length > 0) {
        let minIdx = 0;
        for (let i = 1; i < enrichedWords.length; i++) {
          if ((enrichedWords[i].score ?? 100) < (enrichedWords[minIdx].score ?? 100)) {
            minIdx = i;
          }
        }
        setSelectedWordIdx(minIdx);
      }
    } catch (err: any) {
      setErrorMsg(`Connection error: ${err?.message || 'Unable to reach pronunciation server'}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // --- Computed Display Helpers ---
  const displayWords: WordItem[] = assessResult?.words || previewWords;
  const currentInspectWord: WordItem | undefined =
    displayWords[selectedWordIdx] || displayWords[0];

  const masteredCount = useMemo(
    () => libraryItems.filter((i) => i.best_score >= 85).length,
    [libraryItems]
  );
  const practicedCount = useMemo(
    () => libraryItems.filter((i) => i.attempts_count > 0).length,
    [libraryItems]
  );

  const starsForScore = (score: number) =>
    score >= 90 ? 3 : score >= 75 ? 2 : score >= 55 ? 1 : 0;

  return (
    <>
      {/* =====================================================================
          1. COMPACT STICKY TOP APP BAR & ZERO-OVERFLOW TAB RIBBON
         ===================================================================== */}
      <header
        className="tv-sticky-appbar"
        style={{
          background: isDark ? 'rgba(15, 23, 42, 0.92)' : 'rgba(255, 255, 255, 0.94)',
        }}
      >
        <div className="tv-appbar-inner">
          {/* Compact Brand + Always-Accessible Language Selector */}
          <div className="tv-brand-compact">
            <IconTile tone="navy" size={34} depth={3}>
              <Mic size={17} />
            </IconTile>
            <div style={{ minWidth: 0 }}>
              <div
                style={{
                  fontWeight: 900,
                  fontSize: 16,
                  letterSpacing: '-0.02em',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                <span>TransitVox</span>
                <span
                  title={serverOnline ? 'Connected to Speech Engine' : 'Connecting...'}
                  style={{
                    width: 7,
                    height: 7,
                    borderRadius: 99,
                    backgroundColor: serverOnline ? '#16A34A' : '#F59E0B',
                    display: 'inline-block',
                  }}
                />
              </div>
              <select
                value={selectedLang === 'all' ? 'de' : selectedLang}
                onChange={(e) => {
                  const newLang = e.target.value;
                  setSelectedLang(newLang);
                  const firstForLang = libraryItems.find((i) => i.language === newLang);
                  if (firstForLang) selectLibraryItemForPractice(firstForLang);
                }}
                title="Select Practice & Live STT Language"
                style={{
                  marginTop: 1,
                  padding: '1px 4px',
                  borderRadius: 6,
                  border: `1px solid ${c.border}`,
                  background: c.surface,
                  color: c.text,
                  fontSize: 11,
                  fontWeight: 800,
                  cursor: 'pointer',
                }}
              >
                {LANGUAGE_OPTIONS.map((l) => (
                  <option key={l.code} value={l.code}>
                    {l.flag} {l.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Zero-Overflow 4-Tab Segmented Ribbon */}
          <nav
            className="tv-tab-ribbon"
            style={{
              background: isDark ? '#1E293B' : '#EEF2F6',
              border: `1px solid ${c.border}`,
            }}
          >
            <button
              type="button"
              className="tv-tab-btn"
              onClick={() => setActiveTab('practice')}
              style={{
                background:
                  activeTab === 'practice' ? (isDark ? '#2F5FC4' : '#0D3B84') : 'transparent',
                color: activeTab === 'practice' ? '#FFFFFF' : c.text,
              }}
            >
              <Target size={13} style={{ flexShrink: 0 }} />
              <span>Practice</span>
            </button>
            <button
              type="button"
              className="tv-tab-btn"
              onClick={() => setActiveTab('livestt')}
              style={{
                background:
                  activeTab === 'livestt' ? (isDark ? '#2F5FC4' : '#0D3B84') : 'transparent',
                color: activeTab === 'livestt' ? '#FFFFFF' : c.text,
              }}
            >
              <Mic size={13} style={{ flexShrink: 0 }} />
              <span>Live STT</span>
            </button>
            <button
              type="button"
              className="tv-tab-btn"
              onClick={() => setActiveTab('library')}
              style={{
                background:
                  activeTab === 'library' ? (isDark ? '#2F5FC4' : '#0D3B84') : 'transparent',
                color: activeTab === 'library' ? '#FFFFFF' : c.text,
              }}
            >
              <BookOpen size={13} style={{ flexShrink: 0 }} />
              <span>Library ({filteredItems.length})</span>
            </button>
            <button
              type="button"
              className="tv-tab-btn"
              onClick={() => setActiveTab('progress')}
              style={{
                background:
                  activeTab === 'progress' ? (isDark ? '#2F5FC4' : '#0D3B84') : 'transparent',
                color: activeTab === 'progress' ? '#FFFFFF' : c.text,
              }}
            >
              <TrendingUp size={13} style={{ flexShrink: 0 }} />
              <span>Progress</span>
            </button>
          </nav>

          {/* Compact Right HUD: Streak + Settings */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
            <Pill tone="orange">
              <Flame size={12} /> {streak}
            </Pill>
            <button
              type="button"
              onClick={() => setShowSettingsModal(true)}
              title="Audio & Connection Settings"
              style={{
                width: 34,
                height: 34,
                borderRadius: 10,
                border: `1.5px solid ${c.border}`,
                background: c.surface,
                color: c.text,
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
              }}
            >
              <Settings size={16} />
            </button>
          </div>
        </div>
      </header>

      <div className="tv-app-shell">
        {/* =====================================================================
            TAB 1: PRACTICE STUDIO (USER-CENTRIC PRONUNCIATION WORKSPACE)
           ===================================================================== */}
        {activeTab === 'practice' && (
          <>
            {/* A. Filter & Curriculum Strip (Desktop Inline vs Mobile Popup Trigger) */}
            <GameCard innerStyle={{ padding: '10px 12px' }} style={{ marginBottom: 12 }}>
              {/* Mobile Compact Filter Trigger Row (<768px) */}
              <div className="tv-mobile-filter-bar">
                <button
                  type="button"
                  onClick={() => setShowFilterModal(true)}
                  style={{
                    flex: 1,
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 8,
                    padding: '7px 11px',
                    borderRadius: 10,
                    border: `1.5px solid ${c.border}`,
                    background: c.surface,
                    color: c.text,
                    fontSize: 12,
                    fontWeight: 800,
                    cursor: 'pointer',
                  }}
                >
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                    <SlidersHorizontal size={14} color="#0D3B84" />
                    <span>
                      {LANGUAGE_OPTIONS.find((l) => l.code === selectedLang)?.flag || '🇩🇪'}{' '}
                      {LANGUAGE_OPTIONS.find((l) => l.code === selectedLang)?.label || 'German'} ·{' '}
                      {filterType === 'all' ? 'All' : filterType === 'word' ? 'Words' : 'Phrases'} ·{' '}
                      {filterDifficulty === 'all' ? 'All Levels' : filterDifficulty}
                    </span>
                  </span>
                  <Pill tone="navy">{filteredItems.length}</Pill>
                </button>

                <button
                  type="button"
                  onClick={() => openCreateModal(filterType === 'word' ? 'word' : 'phrase')}
                  style={{
                    padding: '7px 11px',
                    borderRadius: 10,
                    border: 'none',
                    background: '#F89A1C',
                    color: '#1F1300',
                    fontWeight: 900,
                    fontSize: 12,
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 4,
                    cursor: 'pointer',
                    flexShrink: 0,
                  }}
                >
                  <Plus size={14} /> Add
                </button>
              </div>

              {/* Desktop Full Filter Row (>768px) */}
              <div className="tv-desktop-filters">
                <div className="tv-filter-controls">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Globe size={15} color={c.muted} />
                    <select
                      value={selectedLang}
                      onChange={(e) => {
                        const newLang = e.target.value;
                        setSelectedLang(newLang);
                        const firstForLang = libraryItems.find((i) => i.language === newLang);
                        if (firstForLang) selectLibraryItemForPractice(firstForLang);
                      }}
                      style={{
                        padding: '6px 10px',
                        borderRadius: 10,
                        border: `1.5px solid ${c.border}`,
                        background: c.surface,
                        color: c.text,
                        fontWeight: 700,
                        fontSize: 13,
                        cursor: 'pointer',
                      }}
                    >
                      {LANGUAGE_OPTIONS.map((l) => (
                        <option key={l.code} value={l.code}>
                          {l.flag} {l.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Word vs Phrase Mode Filter */}
                  {(['all', 'word', 'phrase'] as const).map((t) => (
                    <button
                      key={t}
                      type="button"
                      onClick={() => setFilterType(t)}
                      style={{
                        padding: '5px 12px',
                        borderRadius: 999,
                        border: `1.5px solid ${filterType === t ? '#0D3B84' : c.border}`,
                        background:
                          filterType === t ? (isDark ? '#2F5FC4' : '#0D3B84') : 'transparent',
                        color: filterType === t ? '#FFFFFF' : c.text,
                        fontWeight: 700,
                        fontSize: 12,
                        cursor: 'pointer',
                      }}
                    >
                      {t === 'all' ? 'All Items' : t === 'word' ? '🔤 Single Words' : '💬 Phrases'}
                    </button>
                  ))}

                  {/* Difficulty Filter */}
                  {(['all', 'Beginner', 'Intermediate', 'Advanced'] as const).map((d) => (
                    <button
                      key={d}
                      type="button"
                      onClick={() => setFilterDifficulty(d)}
                      style={{
                        padding: '5px 10px',
                        borderRadius: 999,
                        border: `1px solid ${filterDifficulty === d ? '#F89A1C' : c.border}`,
                        background:
                          filterDifficulty === d ? 'rgba(248,154,28,0.18)' : 'transparent',
                        color: filterDifficulty === d ? '#F89A1C' : c.muted,
                        fontWeight: 700,
                        fontSize: 12,
                        cursor: 'pointer',
                      }}
                    >
                      {d === 'all' ? 'All Levels' : d}
                    </button>
                  ))}
                </div>

                {/* Add Custom Word/Phrase Button */}
                <ChunkyButton
                  tone="orange"
                  size="sm"
                  onClick={() => openCreateModal(filterType === 'word' ? 'word' : 'phrase')}
                >
                  <Plus size={15} /> Add Word or Phrase
                </ChunkyButton>
              </div>

            {/* Horizontal Swipeable Cards from the DB */}
            <div className="tv-deck-strip">
              {filteredItems.map((item) => {
                const isSelected = item.id === activeItemId;
                const itemStars = starsForScore(item.best_score || 0);
                return (
                  <div
                    key={item.id}
                    className="tv-deck-card"
                    onClick={() => selectLibraryItemForPractice(item)}
                    style={{
                      padding: '10px 12px',
                      borderRadius: 14,
                      border: `2px solid ${
                        isSelected ? '#F89A1C' : c.border
                      }`,
                      background: isSelected
                        ? isDark
                          ? 'rgba(248,154,28,0.14)'
                          : '#FFFBEB'
                        : c.surface,
                      cursor: 'pointer',
                      transition: 'transform 0.15s ease',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: 4,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 10,
                          fontWeight: 800,
                          textTransform: 'uppercase',
                          color: item.item_type === 'word' ? '#8B5CF6' : '#0284C7',
                        }}
                      >
                        {item.item_type === 'word' ? '🔤 Word' : '💬 Phrase'} · {item.difficulty}
                      </span>
                      <StarRow earned={itemStars} total={3} size={12} />
                    </div>
                    <div
                      style={{
                        fontWeight: 800,
                        fontSize: 14,
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        marginBottom: 4,
                      }}
                    >
                      {item.text}
                    </div>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        fontSize: 11,
                        color: c.muted,
                      }}
                    >
                      <span>{item.focus_phoneme || item.category}</span>
                      {item.best_score > 0 && (
                        <strong style={{ color: '#16A34A' }}>Best: {item.best_score}%</strong>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </GameCard>

          {/* B. Main Split Practice Workspace: 3D Coach (Left) + Practice Card (Right) */}
          <div className="gts-stage-grid" style={{ position: 'relative', marginBottom: 16 }}>
            {/* Guided 3-2-1 Countdown Overlay */}
            {countdownValue !== null && (
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  zIndex: 40,
                  borderRadius: 20,
                  background: 'rgba(15, 23, 42, 0.85)',
                  backdropFilter: 'blur(6px)',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#FFFFFF',
                }}
              >
                <div style={{ fontSize: 84, fontWeight: 900, color: '#F89A1C' }}>
                  {countdownValue}
                </div>
                <div style={{ fontSize: 16, fontWeight: 800 }}>
                  Take a breath and get ready to speak...
                </div>
              </div>
            )}

            {/* Left Column: 3D Avatar Coach */}
            <GameCard innerStyle={{ padding: 14 }}>
              <div className="tv-avatar-layout">
                <div
                  className="tv-avatar-header"
                  style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}
                >
                  <select
                    value={characterId}
                    onChange={(e) => setCharacterId(e.target.value)}
                    style={{
                      padding: '5px 10px',
                      borderRadius: 10,
                      border: `1.5px solid ${c.border}`,
                      background: c.surface,
                      color: c.text,
                      fontWeight: 800,
                      fontSize: 12,
                      cursor: 'pointer',
                    }}
                  >
                    {CHARACTERS.map((ch) => (
                      <option key={ch.id} value={ch.id}>
                        {ch.emoji} {ch.label}
                      </option>
                    ))}
                  </select>
                  <Pill tone="purple">{friendlyVisemeLabel(mouthViseme)}</Pill>
                </div>

                <div
                  ref={avatarHostRef}
                  className="tv-avatar-viewport"
                  style={{
                    background: isDark
                      ? 'radial-gradient(circle at 50% 35%, #1e293b, #0f172a)'
                      : 'radial-gradient(circle at 50% 35%, #DBEAFE, #EFF6FF)',
                    border: `2px solid ${isAnalyzing ? '#F89A1C' : c.border}`,
                    transition: 'border-color 0.2s ease',
                  }}
                />

                {isAnalyzing && (
                  <div
                    style={{
                      padding: '6px 10px',
                      borderRadius: 10,
                      background: 'rgba(248,154,28,0.16)',
                      border: '1px solid rgba(248,154,28,0.45)',
                      color: '#F89A1C',
                      fontSize: 12,
                      fontWeight: 800,
                      textAlign: 'center',
                    }}
                  >
                    🤔 {CHARACTERS.find((ch) => ch.id === characterId)?.label || 'Coach'} is thinking...
                  </div>
                )}

                {/* Friendly Coach Speech Bubble */}
                <div
                  style={{
                    padding: '10px 12px',
                    borderRadius: 12,
                    background: isDark ? '#1E293B' : '#F8FAFC',
                    border: `1.5px solid ${c.border}`,
                    fontSize: 13,
                    lineHeight: 1.45,
                  }}
                  dangerouslySetInnerHTML={{ __html: coachBubble }}
                />
              </div>
            </GameCard>

            {/* Right Column: Learner Practice Stage */}
            <GameCard
              style={{
                position: 'relative',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                gap: 14,
              }}
            >
              {/* Thinking / Analyzing Loading Screen Overlay */}
              {isAnalyzing && (
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    zIndex: 35,
                    borderRadius: 20,
                    background: isDark
                      ? 'rgba(15, 23, 42, 0.92)'
                      : 'rgba(255, 255, 255, 0.94)',
                    backdropFilter: 'blur(6px)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: 24,
                    textAlign: 'center',
                    gap: 12,
                  }}
                >
                  <div
                    style={{
                      width: 72,
                      height: 72,
                      borderRadius: 999,
                      background: 'rgba(248, 154, 28, 0.16)',
                      border: '3px solid #F89A1C',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 34,
                      boxShadow: '0 0 24px rgba(248, 154, 28, 0.35)',
                    }}
                  >
                    🤔
                  </div>
                  <div style={{ fontSize: 20, fontWeight: 900 }}>
                    {CHARACTERS.find((ch) => ch.id === characterId)?.label || 'Coach'} is Analyzing Your Voice...
                  </div>
                  <div style={{ fontSize: 13, color: c.muted, maxWidth: 380, lineHeight: 1.45 }}>
                    Checking your mouth shapes, multi-syllable word stress, and smooth word flow for{' '}
                    <strong>“{referenceText}”</strong>
                  </div>
                  <div
                    style={{
                      display: 'flex',
                      gap: 8,
                      flexWrap: 'wrap',
                      justifyContent: 'center',
                      marginTop: 4,
                    }}
                  >
                    <Pill tone="green">✓ Audio Captured</Pill>
                    <Pill tone="orange">⏳ Scoring Sounds & Stress...</Pill>
                  </div>
                </div>
              )}

              {/* Top Bar: Item Metadata + Prev/Next */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: 8,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                  <Pill tone={activeLibraryItem?.item_type === 'word' ? 'purple' : 'blue'}>
                    {activeLibraryItem?.item_type === 'word' ? '🔤 Single Word' : '💬 Phrase'}
                  </Pill>
                  {activeLibraryItem?.category && (
                    <Pill tone="gray">{activeLibraryItem.category}</Pill>
                  )}
                  {activeLibraryItem?.focus_phoneme && (
                    <Pill tone="orange">🎯 Focus: {activeLibraryItem.focus_phoneme}</Pill>
                  )}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <ChunkyButton tone="soft" size="sm" onClick={() => stepDeckItem(-1)}>
                    <ChevronLeft size={15} /> Prev
                  </ChunkyButton>
                  <ChunkyButton tone="soft" size="sm" onClick={() => stepDeckItem(1)}>
                    Next <ChevronRight size={15} />
                  </ChunkyButton>
                </div>
              </div>

              {/* Editable Practice Input (or Quick Custom Sentence) */}
              <div className="tv-input-speed-row">
                <input
                  type="text"
                  value={referenceText}
                  onChange={(e) => {
                    setReferenceText(e.target.value);
                    setAssessResult(null);
                  }}
                  placeholder="Type any word or sentence you want to practice..."
                  style={{
                    flex: 1,
                    minWidth: 0,
                    padding: '9px 12px',
                    borderRadius: 12,
                    border: `1.5px solid ${c.border}`,
                    background: c.surface,
                    color: c.text,
                    fontSize: 14,
                    fontWeight: 700,
                  }}
                />
                <button
                  type="button"
                  onClick={() => setSpeechSpeed((s) => (s === 1.0 ? 0.75 : 1.0))}
                  title="Toggle Normal (1.0x) vs Slow (0.75x) Coach Voice"
                  style={{
                    padding: '8px 12px',
                    borderRadius: 12,
                    border: `1.5px solid ${speechSpeed < 1 ? '#F89A1C' : c.border}`,
                    background: speechSpeed < 1 ? 'rgba(248,154,28,0.16)' : c.surface,
                    color: speechSpeed < 1 ? '#F89A1C' : c.text,
                    fontWeight: 800,
                    fontSize: 12,
                    cursor: 'pointer',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {speechSpeed === 1.0 ? '⚡ Normal 1.0x' : '🐢 Slow 0.75x'}
                </button>
              </div>

              {/* Interactive Word & Syllable Tiles */}
              <div
                className="tv-word-tiles-stage"
                style={{
                  background: isDark ? '#0F172A' : '#F8FAFC',
                  border: `1.5px solid ${c.border}`,
                }}
              >
                {displayWords.map((w, idx) => {
                  const isSel = selectedWordIdx === idx;
                  const isSpk = speakingWordIdx === idx;
                  const statusColor =
                    w.status === 'correct'
                      ? '#16A34A'
                      : w.status === 'warning'
                      ? '#D97706'
                      : w.status === 'mispronounced' || w.status === 'omitted'
                      ? '#DC2626'
                      : isSel
                      ? '#0D3B84'
                      : c.border;

                  const sylList = w.word_stress?.syllables || w.syllables || [];
                  const linkAfter = assessResult?.connected_speech?.links?.[idx];

                  return (
                    <React.Fragment key={`${w.word}-${idx}`}>
                      <div
                        onClick={() => {
                          setSelectedWordIdx(idx);
                          animateWordVisemes(w, 420);
                          if (!w.status) speakCoachPhrase(w.word);
                        }}
                        style={{
                          padding: '10px 14px',
                          borderRadius: 14,
                          border: `2.5px solid ${isSpk ? '#F89A1C' : statusColor}`,
                          background: isSpk
                            ? 'rgba(248,154,28,0.18)'
                            : isSel
                            ? isDark
                              ? '#1E293B'
                              : '#FFFFFF'
                            : c.card,
                          boxShadow: isSel ? '0 4px 12px rgba(13,59,132,0.12)' : 'none',
                          cursor: 'pointer',
                          textAlign: 'center',
                          transition: 'all 0.15s ease',
                        }}
                      >
                        {typeof w.score === 'number' && (
                          <div
                            style={{
                              fontSize: 11,
                              fontWeight: 900,
                              color: statusColor,
                              marginBottom: 2,
                            }}
                          >
                            {w.score}%
                          </div>
                        )}
                        <div style={{ fontSize: 22, fontWeight: 900 }}>{w.word}</div>

                        {/* Syllable Stress Breakdown Pills */}
                        {sylList.length > 1 ? (
                          <div
                            style={{
                              display: 'flex',
                              justifyContent: 'center',
                              gap: 3,
                              marginTop: 4,
                              flexWrap: 'wrap',
                            }}
                          >
                            {sylList.map((s, sIdx) => {
                              const isPrimary = s.stress === 'primary';
                              return (
                                <span
                                  key={sIdx}
                                  style={{
                                    fontSize: 11,
                                    fontFamily: 'monospace',
                                    padding: '1px 6px',
                                    borderRadius: 6,
                                    fontWeight: isPrimary ? 900 : 600,
                                    background: isPrimary
                                      ? 'rgba(139,92,246,0.22)'
                                      : 'transparent',
                                    color: isPrimary ? '#8B5CF6' : c.muted,
                                    border: isPrimary
                                      ? '1px solid rgba(139,92,246,0.5)'
                                      : 'none',
                                  }}
                                >
                                  {isPrimary ? 'ˈ' : ''}
                                  {s.syllable_ipa}
                                </span>
                              );
                            })}
                          </div>
                        ) : (
                          <div
                            style={{
                              fontSize: 12,
                              fontFamily: 'monospace',
                              color: c.muted,
                              marginTop: 3,
                            }}
                          >
                            /{w.expected_ipa}/
                          </div>
                        )}
                      </div>

                      {/* Connected Speech Link Indicator Between Words */}
                      {linkAfter && (
                        <div
                          title={linkAfter.coaching_tip || linkAfter.tip || ''}
                          style={{
                            fontSize: 18,
                            fontWeight: 900,
                            color:
                              linkAfter.status === 'smooth_link' || linkAfter.linked_smoothly
                                ? '#16A34A'
                                : '#D97706',
                            padding: '0 2px',
                            userSelect: 'none',
                          }}
                        >
                          {linkAfter.status === 'smooth_link' || linkAfter.linked_smoothly
                            ? '‿'
                            : '⏸'}
                        </div>
                      )}
                    </React.Fragment>
                  );
                })}
              </div>

              {/* Live Mic Level Bar during Recording */}
              {isRecording && (
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    padding: '8px 12px',
                    borderRadius: 12,
                    background: 'rgba(220,38,38,0.1)',
                    border: '1px solid rgba(220,38,38,0.35)',
                  }}
                >
                  <span style={{ fontSize: 12, fontWeight: 800, color: '#DC2626' }}>
                    🔴 Listening...
                  </span>
                  <div
                    style={{
                      flex: 1,
                      height: 8,
                      borderRadius: 99,
                      background: isDark ? '#1E293B' : '#E2E8F0',
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        width: `${liveMicLevel}%`,
                        height: '100%',
                        background: 'linear-gradient(90deg, #22C55E, #F89A1C)',
                        transition: 'width 0.06s linear',
                      }}
                    />
                  </div>
                  <span style={{ fontSize: 11, color: c.muted }}>
                    {autoStopVad ? 'Auto-stops when you finish' : 'Tap Stop when done'}
                  </span>
                </div>
              )}

              {errorMsg && (
                <div
                  style={{
                    padding: '8px 12px',
                    borderRadius: 10,
                    background: 'rgba(220,38,38,0.12)',
                    color: '#DC2626',
                    fontSize: 13,
                    fontWeight: 700,
                  }}
                >
                  ⚠️ {errorMsg}
                </div>
              )}

              {/* Primary Learner Action Buttons */}
              <div className="tv-action-bar">
                <ChunkyButton
                  tone="soft"
                  onClick={() => speakCoachPhrase()}
                  disabled={isRecording || isAnalyzing}
                >
                  <Volume2 size={17} /> 1. Listen
                </ChunkyButton>

                <div className="tv-speak-main-btn" style={{ display: 'flex' }}>
                  {!isRecording ? (
                    <ChunkyButton
                      tone="orange"
                      fullWidth
                      onClick={startRecording}
                      disabled={isAnalyzing}
                    >
                      <Mic size={17} /> {isAnalyzing ? 'Scoring...' : '2. Speak Now'}
                    </ChunkyButton>
                  ) : (
                    <ChunkyButton tone="red" fullWidth onClick={stopRecordingAndScore}>
                      <Square size={17} /> Stop & Score
                    </ChunkyButton>
                  )}
                </div>

                <ChunkyButton
                  tone="navy"
                  onClick={startGuidedPractice}
                  disabled={isRecording || isAnalyzing}
                >
                  <Sparkles size={16} /> Guided 3-2-1
                </ChunkyButton>

                {recordedAudioUrl && (
                  <ChunkyButton
                    tone="green"
                    onClick={playMyVoice}
                    disabled={isRecording || isPlayingMyVoice}
                  >
                    <Play size={16} /> {isPlayingMyVoice ? 'Playing...' : 'Hear My Voice'}
                  </ChunkyButton>
                )}

                <label
                  className="tv-upload-label"
                  style={{ cursor: 'pointer', marginLeft: 'auto' }}
                  title="Upload a WAV file"
                >
                  <input
                    type="file"
                    accept="audio/wav,audio/*"
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                  />
                  <span
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 5,
                      padding: '8px 12px',
                      borderRadius: 10,
                      border: `1.5px solid ${c.border}`,
                      fontSize: 12,
                      fontWeight: 700,
                      color: c.muted,
                    }}
                  >
                    <Upload size={14} /> Upload Audio
                  </span>
                </label>
              </div>
            </GameCard>
          </div>

          {/* C. Friendly Word, Mouth Placement & Connected Speech Breakdown */}
          {currentInspectWord && (
            <GameCard style={{ padding: '16px 18px' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: 10,
                  marginBottom: 14,
                }}
              >
                <div>
                  <div style={{ fontSize: 18, fontWeight: 900 }}>
                    🔍 Sound & Mouth Guide for “{currentInspectWord.word}”
                    {typeof currentInspectWord.score === 'number' && (
                      <span style={{ marginLeft: 8, color: '#0D3B84' }}>
                        ({currentInspectWord.score}%)
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: 12, color: c.muted }}>
                    Tap any sound card below to see how your mouth and tongue should move
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <ChunkyButton
                    tone="soft"
                    size="sm"
                    onClick={() => speakCoachPhrase(currentInspectWord.word)}
                  >
                    <Volume2 size={14} /> Hear Coach Say “{currentInspectWord.word}”
                  </ChunkyButton>
                  {recordedAudioUrl && typeof currentInspectWord.start_ms === 'number' && (
                    <ChunkyButton
                      tone="green"
                      size="sm"
                      onClick={() => playSelectedWordSlice(currentInspectWord, selectedWordIdx)}
                    >
                      <Play size={14} /> Hear My “{currentInspectWord.word}”
                    </ChunkyButton>
                  )}
                </div>
              </div>

              {/* 1. Multi-Syllable Word Stress Rhythm Bar (if 2+ syllables) */}
              {currentInspectWord.word_stress &&
                currentInspectWord.word_stress.syllable_count > 1 && (
                  <div
                    style={{
                      padding: '12px 14px',
                      borderRadius: 14,
                      background: isDark ? 'rgba(139,92,246,0.12)' : '#F5F3FF',
                      border: '1.5px solid rgba(139,92,246,0.35)',
                      marginBottom: 14,
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                        gap: 8,
                        marginBottom: 8,
                      }}
                    >
                      <span style={{ fontWeight: 900, fontSize: 13, color: '#7C3AED' }}>
                        🎵 Word Stress Rhythm ({currentInspectWord.word_stress.syllable_count}{' '}
                        Syllables)
                      </span>
                      <Pill
                        tone={currentInspectWord.word_stress.stress_correct ? 'green' : 'orange'}
                      >
                        {currentInspectWord.word_stress.stress_correct
                          ? '✅ Natural Stress Rhythm'
                          : '💡 Stress Beat Shifted'}
                      </Pill>
                    </div>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 6 }}>
                      {currentInspectWord.word_stress.syllables.map((s, idx) => {
                        const isPrimary = s.stress === 'primary';
                        return (
                          <div
                            key={idx}
                            style={{
                              padding: '6px 12px',
                              borderRadius: 10,
                              border: isPrimary
                                ? '2px solid #7C3AED'
                                : `1px solid ${c.border}`,
                              background: isPrimary
                                ? 'rgba(139,92,246,0.2)'
                                : c.surface,
                              fontWeight: isPrimary ? 900 : 600,
                              fontSize: 13,
                            }}
                          >
                            /{isPrimary ? 'ˈ' : ''}
                            {s.syllable_ipa}/ ·{' '}
                            <span style={{ fontSize: 11, color: c.muted }}>
                              {isPrimary ? 'STRESSED (Longer & Louder)' : 'Relaxed'}
                              {s.duration_ms ? ` · ${s.duration_ms}ms` : ''}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                    <div style={{ fontSize: 12, color: c.text }}>
                      {currentInspectWord.word_stress.coaching_tip ||
                        currentInspectWord.word_stress.tip ||
                        'Make the stressed syllable slightly longer and higher in pitch.'}
                    </div>
                  </div>
                )}

              {/* 2. Sound-by-Sound Cards with Plain-English Articulatory Diff */}
              <div
                className="tv-phoneme-grid"
                style={{
                  marginBottom: assessResult?.connected_speech ? 14 : 0,
                }}
              >
                {(currentInspectWord.phonemes || []).map((p, pIdx) => {
                  const isStr = typeof p === 'string';
                  const expectedSym = isStr ? p : p.expected || '';
                  const heardSym = isStr ? p : p.heard || '∅';
                  const score = isStr ? null : p.gop_score;
                  const hasErr = !isStr && p.error !== null;
                  const guide = FRIENDLY_PHONEME_GUIDES[expectedSym];
                  const diffTip = !isStr ? p.articulatory_diff : null;

                  return (
                    <div
                      key={pIdx}
                      onClick={() => setAvatarViseme(ipaToViseme(expectedSym))}
                      style={{
                        padding: '10px 12px',
                        borderRadius: 12,
                        border: `1.5px solid ${
                          score === null
                            ? c.border
                            : !hasErr
                            ? 'rgba(22,163,74,0.45)'
                            : 'rgba(220,38,38,0.55)'
                        }`,
                        background:
                          score === null
                            ? c.surface
                            : !hasErr
                            ? isDark
                              ? 'rgba(22,163,74,0.1)'
                              : '#F0FDF4'
                            : isDark
                            ? 'rgba(220,38,38,0.12)'
                            : '#FEF2F2',
                        cursor: 'pointer',
                      }}
                    >
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          marginBottom: 4,
                        }}
                      >
                        <span style={{ fontFamily: 'monospace', fontSize: 16, fontWeight: 900 }}>
                          /{expectedSym}/ {hasErr ? `→ /${heardSym}/` : ''}
                        </span>
                        {score !== null && (
                          <span
                            style={{
                              fontSize: 12,
                              fontWeight: 900,
                              color: !hasErr ? '#16A34A' : '#DC2626',
                            }}
                          >
                            {!hasErr ? `✅ ${score}%` : `⚠️ ${score}%`}
                          </span>
                        )}
                      </div>

                      {guide && (
                        <div style={{ fontSize: 11, fontWeight: 800, color: '#0D3B84', marginBottom: 3 }}>
                          {guide.name} (as in “{guide.example}”)
                        </div>
                      )}

                      <div style={{ fontSize: 12, color: c.text, lineHeight: 1.4 }}>
                        {diffTip || guide?.howTo || 'Tap to preview mouth shape on the 3D Coach.'}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* 3. Connected Speech Summary (for Multi-Word Phrases) */}
              {assessResult?.connected_speech &&
                (assessResult.connected_speech.links || []).length > 0 && (
                  <div
                    style={{
                      padding: '12px 14px',
                      borderRadius: 14,
                      background: isDark ? 'rgba(14,165,233,0.1)' : '#F0F9FF',
                      border: '1.5px solid rgba(14,165,233,0.35)',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                        gap: 8,
                        marginBottom: 6,
                      }}
                    >
                      <span style={{ fontWeight: 900, fontSize: 13, color: '#0284C7' }}>
                        🌊 Connected Speech & Word Flow (
                        {assessResult.connected_speech.linking_score}%)
                      </span>
                      <Pill tone="blue">
                        Melody: {assessResult.connected_speech.intonation_type.replace('_', ' ')}
                      </Pill>
                    </div>
                    <div style={{ fontSize: 12, marginBottom: 8 }}>
                      {assessResult.connected_speech.summary_tip}
                    </div>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {assessResult.connected_speech.links.map((lnk, idx) => {
                        const smooth =
                          lnk.status === 'smooth_link' || Boolean(lnk.linked_smoothly);
                        return (
                          <div
                            key={idx}
                            style={{
                              padding: '6px 10px',
                              borderRadius: 10,
                              background: c.surface,
                              border: `1px solid ${smooth ? '#16A34A' : '#D97706'}`,
                              fontSize: 12,
                            }}
                          >
                            <strong>
                              {lnk.from_word} {smooth ? '‿' : '⏸'} {lnk.to_word}
                            </strong>{' '}
                            <span style={{ color: c.muted }}>
                              ({smooth ? 'Smooth link' : `${lnk.gap_ms}ms pause`})
                            </span>
                            {(lnk.coaching_tip || lnk.tip) && (
                              <div style={{ fontSize: 11, color: c.muted, marginTop: 2 }}>
                                {lnk.coaching_tip || lnk.tip}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
            </GameCard>
          )}
        </>
      )}

      {/* =====================================================================
          TAB 1B: LIVE STT & FREE-SPEECH PRONUNCIATION COACH (faster-whisper-small)
         ===================================================================== */}
      {activeTab === 'livestt' && (
        <GameCard innerStyle={{ padding: '18px 20px' }} style={{ position: 'relative' }}>
          {isAnalyzing && (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                zIndex: 35,
                borderRadius: 20,
                background: isDark ? 'rgba(15, 23, 42, 0.92)' : 'rgba(255, 255, 255, 0.94)',
                backdropFilter: 'blur(6px)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 24,
                textAlign: 'center',
                gap: 12,
              }}
            >
              <div
                style={{
                  width: 68,
                  height: 68,
                  borderRadius: 999,
                  background: 'rgba(248, 154, 28, 0.16)',
                  border: '3px solid #F89A1C',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 32,
                }}
              >
                🤔
              </div>
              <div style={{ fontSize: 19, fontWeight: 900 }}>
                Transcribing & Scoring Your Free Speech...
              </div>
              <div style={{ fontSize: 13, color: c.muted, maxWidth: 380 }}>
                Running <strong>faster-whisper-small (INT8)</strong> in{' '}
                {LANGUAGE_OPTIONS.find((l) => l.code === selectedLang)?.label || 'German'} and scoring
                every phoneme & syllable stress beat
              </div>
            </div>
          )}

          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: 10,
              marginBottom: 14,
            }}
          >
            <div>
              <div style={{ fontSize: 19, fontWeight: 900 }}>
                🎙️ Live Speech-to-Text & Free-Speak Coach
              </div>
              <div style={{ fontSize: 12.5, color: c.muted }}>
                Speak freely in your chosen language — watch live transcription appear and score your pronunciation
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <Pill tone={modeAReady ? 'green' : 'orange'}>
                {modeAReady ? '🟢 faster-whisper-small Ready' : '⚡ Whisper STT'}
              </Pill>
              <select
                value={selectedLang === 'all' ? 'de' : selectedLang}
                onChange={(e) => setSelectedLang(e.target.value)}
                style={{
                  padding: '7px 11px',
                  borderRadius: 10,
                  border: `1.5px solid ${c.border}`,
                  background: c.surface,
                  color: c.text,
                  fontWeight: 800,
                  fontSize: 13,
                  cursor: 'pointer',
                }}
              >
                {LANGUAGE_OPTIONS.map((l) => (
                  <option key={l.code} value={l.code}>
                    {l.flag} {l.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Live Transcript Display Box */}
          <div
            style={{
              minHeight: 135,
              padding: '16px 18px',
              borderRadius: 16,
              background: isDark ? '#0F172A' : '#F8FAFC',
              border: `2px solid ${isLiveSttStreaming ? '#F89A1C' : c.border}`,
              marginBottom: 14,
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              gap: 10,
            }}
          >
            <div style={{ fontSize: 18, fontWeight: 800, lineHeight: 1.5, color: c.text }}>
              {liveSttFinalList.length === 0 && !liveSttInterim ? (
                <span style={{ color: c.muted, fontWeight: 600, fontSize: 15 }}>
                  {isLiveSttStreaming
                    ? `👂 Listening in ${
                        LANGUAGE_OPTIONS.find((l) => l.code === selectedLang)?.label || 'German'
                      }... Speak any sentence!`
                    : 'Tap "Start Live Speaking" below and say anything in your selected language...'}
                </span>
              ) : (
                <>
                  <span>{liveSttFinalList.join(' ')} </span>
                  {liveSttInterim && (
                    <span style={{ color: '#F89A1C', fontStyle: 'italic' }}>
                      {liveSttInterim} ...
                    </span>
                  )}
                </>
              )}
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: 8,
                fontSize: 11.5,
                color: c.muted,
              }}
            >
              <span>
                Language:{' '}
                <strong>
                  {LANGUAGE_OPTIONS.find((l) => l.code === selectedLang)?.flag || '🇩🇪'}{' '}
                  {LANGUAGE_OPTIONS.find((l) => l.code === selectedLang)?.label || 'German'}
                </strong>
                {liveSttConfidence !== null && (
                  <> · STT Confidence: <strong>{Math.round(liveSttConfidence * 100)}%</strong></>
                )}
              </span>
              {(liveSttFinalList.length > 0 || liveSttInterim) && (
                <button
                  type="button"
                  onClick={() => {
                    setLiveSttFinalList([]);
                    setLiveSttInterim('');
                    setLiveSttConfidence(null);
                  }}
                  style={{
                    border: 'none',
                    background: 'transparent',
                    color: '#DC2626',
                    fontWeight: 800,
                    fontSize: 11.5,
                    cursor: 'pointer',
                  }}
                >
                  Clear Transcript
                </button>
              )}
            </div>
          </div>

          {(isLiveSttStreaming || isSttRecordingClip) && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '8px 12px',
                borderRadius: 12,
                background: 'rgba(248,154,28,0.12)',
                border: '1px solid rgba(248,154,28,0.4)',
                marginBottom: 12,
              }}
            >
              <span style={{ fontSize: 12, fontWeight: 800, color: '#F89A1C' }}>
                {isSttRecordingClip ? '🔴 Recording Clip...' : '🔴 Live Streaming...'}
              </span>
              <div
                style={{
                  flex: 1,
                  height: 8,
                  borderRadius: 99,
                  background: isDark ? '#1E293B' : '#E2E8F0',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    width: `${liveMicLevel}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, #22C55E, #F89A1C)',
                    transition: 'width 0.06s linear',
                  }}
                />
              </div>
            </div>
          )}

          {errorMsg && (
            <div
              style={{
                padding: '8px 12px',
                borderRadius: 10,
                background: 'rgba(220,38,38,0.12)',
                color: '#DC2626',
                fontSize: 13,
                fontWeight: 700,
                marginBottom: 12,
              }}
            >
              ⚠️ {errorMsg}
            </div>
          )}

          <div className="tv-action-bar">
            {!isLiveSttStreaming && !isSttRecordingClip && (
              <>
                <div className="tv-speak-main-btn" style={{ display: 'flex' }}>
                  <ChunkyButton
                    tone="orange"
                    fullWidth
                    onClick={startSttClipRecording}
                    disabled={isAnalyzing}
                  >
                    <Mic size={17} /> Record to Transcribe
                  </ChunkyButton>
                </div>

                <ChunkyButton
                  tone="navy"
                  onClick={startLiveSttStream}
                  disabled={isAnalyzing}
                >
                  <Sparkles size={16} /> Live Stream STT
                </ChunkyButton>
              </>
            )}

            {isSttRecordingClip && (
              <>
                <div className="tv-speak-main-btn" style={{ display: 'flex' }}>
                  <ChunkyButton
                    tone="navy"
                    fullWidth
                    onClick={() => stopSttClipRecording(false)}
                  >
                    <Square size={16} /> Stop & Transcribe
                  </ChunkyButton>
                </div>
                <ChunkyButton
                  tone="green"
                  onClick={() => stopSttClipRecording(true)}
                >
                  <CheckCircle2 size={16} /> Stop, Transcribe & Score
                </ChunkyButton>
              </>
            )}

            {isLiveSttStreaming && (
              <>
                <div className="tv-speak-main-btn" style={{ display: 'flex' }}>
                  <ChunkyButton
                    tone="green"
                    fullWidth
                    onClick={() => stopLiveSttStream(true)}
                  >
                    <CheckCircle2 size={17} /> Stop & Score Pronunciation
                  </ChunkyButton>
                </div>
                <ChunkyButton tone="red" onClick={() => stopLiveSttStream(false)}>
                  <Square size={16} /> Stop Stream
                </ChunkyButton>
              </>
            )}

            {/* Upload Audio to Transcribe Only */}
            {!isLiveSttStreaming && !isSttRecordingClip && (
              <label style={{ cursor: 'pointer' }} title="Upload WAV, MP3, M4A, or WebM to Transcribe">
                <input
                  type="file"
                  accept="audio/*"
                  onChange={(e) => handleSttFileUpload(e, false)}
                  disabled={isAnalyzing}
                  style={{ display: 'none' }}
                />
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 6,
                    height: 46,
                    padding: '0 16px',
                    borderRadius: 14,
                    border: `2px solid ${c.border}`,
                    background: c.face,
                    color: c.text,
                    fontSize: 13,
                    fontWeight: 900,
                  }}
                >
                  <Upload size={15} /> Upload to Transcribe
                </span>
              </label>
            )}

            {/* Upload Audio to Transcribe & Score Pronunciation */}
            {!isLiveSttStreaming && !isSttRecordingClip && (
              <label
                style={{ cursor: 'pointer' }}
                title="Upload audio to transcribe with Whisper AND score pronunciation"
              >
                <input
                  type="file"
                  accept="audio/*"
                  onChange={(e) => handleSttFileUpload(e, true)}
                  disabled={isAnalyzing}
                  style={{ display: 'none' }}
                />
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 6,
                    height: 46,
                    padding: '0 16px',
                    borderRadius: 14,
                    border: '2px solid #0D3B84',
                    background: isDark ? '#2F5FC4' : '#0D3B84',
                    color: '#FFFFFF',
                    fontSize: 13,
                    fontWeight: 900,
                  }}
                >
                  <Sparkles size={15} /> Upload & Score
                </span>
              </label>
            )}

            {recordedAudioUrl && !isLiveSttStreaming && !isSttRecordingClip && (
              <ChunkyButton
                tone="green"
                onClick={playMyVoice}
                disabled={isPlayingMyVoice}
              >
                <Play size={15} /> {isPlayingMyVoice ? 'Playing...' : 'Play Audio'}
              </ChunkyButton>
            )}

            {(liveSttFinalList.length > 0 || liveSttInterim) &&
              !isLiveSttStreaming &&
              !isSttRecordingClip && (
                <ChunkyButton
                  tone="soft"
                  onClick={() => {
                    const text = [...liveSttFinalList, liveSttInterim].join(' ').trim();
                    if (text) {
                      setReferenceText(text);
                      setActiveTab('practice');
                    }
                  }}
                >
                  <Target size={15} /> Open in Practice Studio
                </ChunkyButton>
              )}
          </div>
        </GameCard>
      )}

      {/* =====================================================================
          TAB 2: WORD & PHRASE BANK (SQLITE DATABASE MANAGER)
         ===================================================================== */}
      {activeTab === 'library' && (
        <GameCard style={{ padding: '18px 20px' }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: 12,
              marginBottom: 16,
            }}
          >
            <div>
              <div style={{ fontSize: 20, fontWeight: 900 }}>
                📚 Word & Phrase Bank
              </div>
              <div style={{ fontSize: 13, color: c.muted }}>
                Browse, search, or add your own words and sentences with automatic IPA & syllable stress breakdown
              </div>
            </div>

            <div style={{ display: 'flex', gap: 8 }}>
              <ChunkyButton tone="navy" size="sm" onClick={() => openCreateModal('word')}>
                <Plus size={15} /> Add Single Word
              </ChunkyButton>
              <ChunkyButton tone="orange" size="sm" onClick={() => openCreateModal('phrase')}>
                <Plus size={15} /> Add Phrase / Sentence
              </ChunkyButton>
            </div>
          </div>

          {/* Search & Filters Row */}
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 10,
              marginBottom: 16,
              alignItems: 'center',
            }}
          >
            <div style={{ position: 'relative', flex: '1 1 240px' }}>
              <Search
                size={15}
                color={c.muted}
                style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)' }}
              />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search words, phrases, categories, or sounds (e.g. /θ/, interview)..."
                style={{
                  width: '100%',
                  padding: '8px 12px 8px 34px',
                  borderRadius: 10,
                  border: `1.5px solid ${c.border}`,
                  background: c.surface,
                  color: c.text,
                  fontSize: 13,
                }}
              />
            </div>

            <select
              value={selectedLang}
              onChange={(e) => setSelectedLang(e.target.value)}
              style={{
                padding: '8px 10px',
                borderRadius: 10,
                border: `1.5px solid ${c.border}`,
                background: c.surface,
                color: c.text,
                fontWeight: 700,
                fontSize: 13,
              }}
            >
              <option value="all">🌐 All Languages</option>
              {LANGUAGE_OPTIONS.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.flag} {l.label}
                </option>
              ))}
            </select>

            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as any)}
              style={{
                padding: '8px 10px',
                borderRadius: 10,
                border: `1.5px solid ${c.border}`,
                background: c.surface,
                color: c.text,
                fontWeight: 700,
                fontSize: 13,
              }}
            >
              <option value="all">All Types (Words & Phrases)</option>
              <option value="word">🔤 Single Words Only</option>
              <option value="phrase">💬 Phrases & Sentences Only</option>
            </select>

            <select
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value)}
              style={{
                padding: '8px 10px',
                borderRadius: 10,
                border: `1.5px solid ${c.border}`,
                background: c.surface,
                color: c.text,
                fontWeight: 700,
                fontSize: 13,
              }}
            >
              <option value="all">All Categories</option>
              {CATEGORY_OPTIONS.map((cat) => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>
          </div>

          {/* Items Grid */}
          <div className="tv-library-grid">
            {filteredItems.map((item) => {
              const langMeta = LANGUAGE_OPTIONS.find((l) => l.code === item.language);
              return (
                <div
                  key={item.id}
                  style={{
                    padding: '14px 16px',
                    borderRadius: 14,
                    border: `1.5px solid ${c.border}`,
                    background: c.surface,
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    gap: 10,
                  }}
                >
                  <div>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: 6,
                      }}
                    >
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
                        <Pill tone={item.item_type === 'word' ? 'purple' : 'blue'}>
                          {item.item_type === 'word' ? '🔤 Word' : '💬 Phrase'}
                        </Pill>
                        <Pill tone="gray">
                          {langMeta?.flag || '🌐'} {item.difficulty}
                        </Pill>
                        {item.focus_phoneme && (
                          <Pill tone="orange">{item.focus_phoneme}</Pill>
                        )}
                      </div>

                      <div style={{ display: 'flex', gap: 4 }}>
                        <button
                          type="button"
                          onClick={() => openEditModal(item)}
                          title="Edit"
                          style={{
                            border: 'none',
                            background: 'transparent',
                            color: c.muted,
                            cursor: 'pointer',
                            padding: 4,
                          }}
                        >
                          <Edit3 size={14} />
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDeleteLibraryItem(item.id)}
                          title="Delete"
                          style={{
                            border: 'none',
                            background: 'transparent',
                            color: '#DC2626',
                            cursor: 'pointer',
                            padding: 4,
                          }}
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </div>

                    <div style={{ fontSize: 17, fontWeight: 900, marginBottom: 3 }}>
                      {item.text}
                    </div>
                    {item.expected_ipa && (
                      <div
                        style={{
                          fontSize: 12,
                          fontFamily: 'monospace',
                          color: '#0D3B84',
                          marginBottom: 6,
                        }}
                      >
                        /{item.expected_ipa}/
                      </div>
                    )}
                    {item.meaning_or_tip && (
                      <div style={{ fontSize: 12, color: c.muted, lineHeight: 1.4 }}>
                        💡 {item.meaning_or_tip}
                      </div>
                    )}
                  </div>

                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      paddingTop: 8,
                      borderTop: `1px solid ${c.border}`,
                    }}
                  >
                    <div style={{ fontSize: 12, color: c.muted }}>
                      {item.attempts_count > 0 ? (
                        <span>
                          Best: <strong style={{ color: '#16A34A' }}>{item.best_score}%</strong> ·{' '}
                          {item.attempts_count} tries
                        </span>
                      ) : (
                        <span>Not practiced yet</span>
                      )}
                    </div>
                    <ChunkyButton
                      tone="navy"
                      size="sm"
                      onClick={() => selectLibraryItemForPractice(item)}
                    >
                      <Target size={14} /> Practice
                    </ChunkyButton>
                  </div>
                </div>
              );
            })}
          </div>
        </GameCard>
      )}

      {/* =====================================================================
          TAB 3: MY PROGRESS & WEAK SOUNDS COACH
         ===================================================================== */}
      {activeTab === 'progress' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="tv-progress-stats-grid">
            <StatTile
              label="Mastered (85%+)"
              value={`${libraryStats?.mastered_count ?? masteredCount} / ${libraryItems.length}`}
              tone="green"
            />
            <StatTile
              label="Items Practiced"
              value={`${libraryStats?.practiced_count ?? practicedCount}`}
              tone="blue"
            />
            <StatTile
              label="Average Score"
              value={`${libraryStats?.average_score || 82}%`}
              tone="navy"
            />
            <StatTile label="Total XP" value={`${xp} XP`} tone="orange" />
          </div>

          {/* Sounds to Focus On */}
          <GameCard style={{ padding: '18px 20px' }}>
            <div style={{ fontSize: 18, fontWeight: 900, marginBottom: 4 }}>
              🎯 Sounds & Mouth Shapes Guide
            </div>
            <div style={{ fontSize: 13, color: c.muted, marginBottom: 14 }}>
              Click any sound below to filter practice words that train that exact mouth placement
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
                gap: 12,
              }}
            >
              {Object.entries(FRIENDLY_PHONEME_GUIDES).map(([sym, info]) => (
                <div
                  key={sym}
                  style={{
                    padding: '12px 14px',
                    borderRadius: 14,
                    border: `1.5px solid ${c.border}`,
                    background: c.surface,
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    gap: 8,
                  }}
                >
                  <div>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: 4,
                      }}
                    >
                      <span style={{ fontSize: 18, fontWeight: 900, fontFamily: 'monospace' }}>
                        /{sym}/ · {info.name}
                      </span>
                      <Pill tone="purple">{info.mouthShape}</Pill>
                    </div>
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#F89A1C', marginBottom: 4 }}>
                      Examples: {info.example}
                    </div>
                    <div style={{ fontSize: 12, color: c.muted, lineHeight: 1.4 }}>
                      {info.howTo}
                    </div>
                  </div>
                  <ChunkyButton
                    tone="soft"
                    size="sm"
                    onClick={() => {
                      const match = libraryItems.find(
                        (it) =>
                          it.focus_phoneme.includes(sym) ||
                          it.expected_ipa.includes(sym)
                      );
                      if (match) selectLibraryItemForPractice(match);
                      else setActiveTab('practice');
                    }}
                  >
                    <Target size={13} /> Practice /{sym}/ Words
                  </ChunkyButton>
                </div>
              ))}
            </div>
          </GameCard>
        </div>
      )}

      {/* =====================================================================
          MODAL 1: RESULT POPUP MODAL AFTER SPEAKING
         ===================================================================== */}
      {showResultPopup && assessResult && (
        <div
          className="tv-modal-backdrop"
          onClick={() => setShowResultPopup(false)}
        >
          <div
            className="tv-modal-scroll-box"
            onClick={(e) => e.stopPropagation()}
            style={{ maxWidth: 500 }}
          >
            <GameCard innerStyle={{ padding: '18px 20px', textAlign: 'center' }}>
              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button
                  type="button"
                  onClick={() => setShowResultPopup(false)}
                  style={{
                    border: 'none',
                    background: 'transparent',
                    color: c.muted,
                    cursor: 'pointer',
                  }}
                >
                  <X size={20} />
                </button>
              </div>

              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 8 }}>
                <StarRow earned={starsForScore(assessResult.overall_score)} total={3} size={28} />
              </div>

              <div style={{ fontSize: 20, fontWeight: 900, marginBottom: 4 }}>
                {assessResult.overall_score >= 90
                  ? '🌟 Outstanding Pronunciation!'
                  : assessResult.overall_score >= 75
                  ? '👏 Great Job — Very Clear!'
                  : '💪 Good Effort — Let’s Polish It!'}
              </div>
              <div style={{ fontSize: 13, color: c.muted, marginBottom: 14 }}>
                “{referenceText}”
              </div>

              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 16 }}>
                <ProgressRing
                  value={assessResult.overall_score}
                  size={108}
                  stroke={10}
                  tone={
                    assessResult.overall_score >= 85
                      ? 'green'
                      : assessResult.overall_score >= 65
                      ? 'orange'
                      : 'red'
                  }
                  label={`${assessResult.overall_score}%`}
                  sublabel="Overall"
                />
              </div>

              <div className="tv-popup-stats-grid">
                <StatTile
                  label="Sounds"
                  value={`${assessResult.metrics.accuracy}%`}
                  tone="green"
                />
                <StatTile
                  label="Smoothness"
                  value={`${assessResult.metrics.fluency}%`}
                  tone="blue"
                />
                <StatTile
                  label="Rhythm"
                  value={`${assessResult.metrics.prosody_similarity}%`}
                  tone="purple"
                />
              </div>

              {/* Short Analysis Box inside Popup */}
              {(() => {
                const words = assessResult.words || [];
                let topSoundIssue: { word: string; expected: string; heard: string; tip: string } | null = null;
                for (const w of words) {
                  for (const p of w.phonemes || []) {
                    if (typeof p !== 'string' && p.error && p.expected) {
                      const guide = FRIENDLY_PHONEME_GUIDES[p.expected];
                      topSoundIssue = {
                        word: w.word,
                        expected: p.expected,
                        heard: p.heard || '∅',
                        tip:
                          p.articulatory_diff ||
                          guide?.howTo ||
                          `Focus on the /${p.expected}/ sound in "${w.word}".`,
                      };
                      break;
                    }
                  }
                  if (topSoundIssue) break;
                }

                const stressShiftWord = words.find(
                  (w) =>
                    w.word_stress &&
                    w.word_stress.syllable_count > 1 &&
                    !w.word_stress.stress_correct
                );

                return (
                  <div
                    style={{
                      textAlign: 'left',
                      padding: '12px 14px',
                      borderRadius: 14,
                      background: isDark ? '#1E293B' : '#F8FAFC',
                      border: `1.5px solid ${c.border}`,
                      marginBottom: 16,
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 8,
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                        gap: 6,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 11,
                          fontWeight: 900,
                          textTransform: 'uppercase',
                          letterSpacing: 0.4,
                          color: '#0D3B84',
                        }}
                      >
                        📋 Quick Coach Summary
                      </span>
                      <span style={{ fontSize: 10.5, color: c.muted, fontWeight: 700 }}>
                        {isPlayingMyVoice
                          ? '🔊 Playing your voice...'
                          : '👆 Tap any word to hear your recording'}
                      </span>
                    </div>

                    {/* Interactive Word-by-Word Playback Strip with Live Speaking Animation */}
                    <div
                      style={{
                        display: 'flex',
                        flexWrap: 'wrap',
                        gap: 7,
                        alignItems: 'center',
                        padding: '6px 0 2px',
                      }}
                    >
                      {words.map((w, idx) => {
                        const ok = w.status === 'correct';
                        const warn = w.status === 'warning';
                        const isWordPlaying = speakingWordIdx === idx;
                        const bg = isWordPlaying
                          ? '#F89A1C'
                          : ok
                          ? 'rgba(22,163,74,0.14)'
                          : warn
                          ? 'rgba(217,119,6,0.15)'
                          : 'rgba(220,38,38,0.15)';
                        const fg = isWordPlaying
                          ? '#1F1300'
                          : ok
                          ? '#16A34A'
                          : warn
                          ? '#D97706'
                          : '#DC2626';
                        return (
                          <button
                            key={idx}
                            type="button"
                            className={isWordPlaying ? 'tv-speaking-word-Active' : undefined}
                            onClick={() => playSelectedWordSlice(w, idx)}
                            title={`Tap to hear your recording of "${w.word}"`}
                            style={{
                              padding: '5px 11px',
                              borderRadius: 999,
                              border: isWordPlaying
                                ? '2px solid #B86E0B'
                                : `1.5px solid ${fg}44`,
                              background: bg,
                              color: fg,
                              fontWeight: 900,
                              fontSize: 12.5,
                              cursor: 'pointer',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: 4,
                              transition: 'all 0.15s ease',
                            }}
                          >
                            {isWordPlaying && (
                              <span className="tv-eq-bars">
                                <span />
                                <span />
                                <span />
                              </span>
                            )}
                            <span>{w.word}</span>
                            {typeof w.score === 'number' && (
                              <span style={{ opacity: 0.85, fontSize: 11 }}>{w.score}%</span>
                            )}
                          </button>
                        );
                      })}
                    </div>

                    {/* Top Sound Takeaway */}
                    <div style={{ fontSize: 12, lineHeight: 1.45, color: c.text }}>
                      {topSoundIssue ? (
                        <>
                          👄 <strong>Sound Fix ({topSoundIssue.word}):</strong> You said{' '}
                          <code>/{topSoundIssue.heard}/</code> instead of{' '}
                          <code>/{topSoundIssue.expected}/</code> — {topSoundIssue.tip}
                        </>
                      ) : (
                        <>
                          ✅ <strong>Sounds:</strong> Every target sound matched clearly!
                        </>
                      )}
                    </div>

                    {/* Stress or Connected Speech Takeaway */}
                    {stressShiftWord && stressShiftWord.word_stress ? (
                      <div style={{ fontSize: 12, lineHeight: 1.45, color: c.text }}>
                        🎵 <strong>Word Stress ({stressShiftWord.word}):</strong>{' '}
                        {stressShiftWord.word_stress.coaching_tip ||
                          stressShiftWord.word_stress.tip ||
                          'Make the stressed syllable slightly longer and louder.'}
                      </div>
                    ) : assessResult.connected_speech?.summary_tip ? (
                      <div style={{ fontSize: 12, lineHeight: 1.45, color: c.text }}>
                        🌊 <strong>Flow:</strong> {assessResult.connected_speech.summary_tip}
                      </div>
                    ) : null}
                  </div>
                );
              })()}

              <div className="tv-popup-actions">
                {recordedAudioUrl && (
                  <ChunkyButton tone="green" size="sm" onClick={playMyVoice}>
                    <Play size={15} /> Hear My Voice
                  </ChunkyButton>
                )}
                <ChunkyButton
                  tone="orange"
                  size="sm"
                  onClick={() => {
                    setShowResultPopup(false);
                    startRecording();
                  }}
                >
                  <Mic size={15} /> Try Again
                </ChunkyButton>
                <ChunkyButton
                  tone="soft"
                  size="sm"
                  onClick={() => setShowResultPopup(false)}
                >
                  <CheckCircle2 size={15} /> Full Breakdown
                </ChunkyButton>
                <ChunkyButton
                  tone="navy"
                  size="sm"
                  onClick={() => {
                    setShowResultPopup(false);
                    stepDeckItem(1);
                  }}
                >
                  Next <ChevronRight size={15} />
                </ChunkyButton>
              </div>
            </GameCard>
          </div>
        </div>
      )}

      {/* =====================================================================
          MODAL 2: ADD / EDIT WORD OR PHRASE IN DATABASE
         ===================================================================== */}
      {showItemModal && (
        <div
          className="tv-modal-backdrop"
          onClick={() => setShowItemModal(false)}
        >
          <div
            className="tv-modal-scroll-box"
            onClick={(e) => e.stopPropagation()}
            style={{ maxWidth: 520 }}
          >
            <GameCard innerStyle={{ padding: '18px 20px' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 14,
                }}
              >
                <div style={{ fontSize: 18, fontWeight: 900 }}>
                  {editingItem ? '✏️ Edit Word or Phrase' : '➕ Add Word or Phrase to Library'}
                </div>
                <button
                  type="button"
                  onClick={() => setShowItemModal(false)}
                  style={{
                    border: 'none',
                    background: 'transparent',
                    color: c.muted,
                    cursor: 'pointer',
                  }}
                >
                  <X size={20} />
                </button>
              </div>

              <form onSubmit={handleSaveLibraryItem} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div>
                  <label style={{ fontSize: 12, fontWeight: 800, display: 'block', marginBottom: 4 }}>
                    Word or Phrase Text *
                  </label>
                  <input
                    type="text"
                    required
                    value={formText}
                    onChange={(e) => setFormText(e.target.value)}
                    placeholder="e.g. Collaboration or Could you repeat that please"
                    style={{
                      width: '100%',
                      padding: '9px 12px',
                      borderRadius: 10,
                      border: `1.5px solid ${c.border}`,
                      background: c.surface,
                      color: c.text,
                      fontSize: 14,
                      fontWeight: 700,
                    }}
                  />
                </div>

                <div className="tv-form-two-col">
                  <div>
                    <label style={{ fontSize: 12, fontWeight: 800, display: 'block', marginBottom: 4 }}>
                      Language
                    </label>
                    <select
                      value={formLang}
                      onChange={(e) => setFormLang(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '8px 10px',
                        borderRadius: 10,
                        border: `1.5px solid ${c.border}`,
                        background: c.surface,
                        color: c.text,
                        fontWeight: 700,
                      }}
                    >
                      {LANGUAGE_OPTIONS.map((l) => (
                        <option key={l.code} value={l.code}>
                          {l.flag} {l.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label style={{ fontSize: 12, fontWeight: 800, display: 'block', marginBottom: 4 }}>
                      Difficulty
                    </label>
                    <select
                      value={formDifficulty}
                      onChange={(e) => setFormDifficulty(e.target.value as any)}
                      style={{
                        width: '100%',
                        padding: '8px 10px',
                        borderRadius: 10,
                        border: `1.5px solid ${c.border}`,
                        background: c.surface,
                        color: c.text,
                        fontWeight: 700,
                      }}
                    >
                      <option value="Beginner">🌱 Beginner</option>
                      <option value="Intermediate">⚡ Intermediate</option>
                      <option value="Advanced">🔥 Advanced</option>
                    </select>
                  </div>
                </div>

                <div className="tv-form-two-col">
                  <div>
                    <label style={{ fontSize: 12, fontWeight: 800, display: 'block', marginBottom: 4 }}>
                      Category
                    </label>
                    <select
                      value={formCategory}
                      onChange={(e) => setFormCategory(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '8px 10px',
                        borderRadius: 10,
                        border: `1.5px solid ${c.border}`,
                        background: c.surface,
                        color: c.text,
                        fontWeight: 700,
                      }}
                    >
                      {CATEGORY_OPTIONS.map((cat) => (
                        <option key={cat} value={cat}>
                          {cat}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label style={{ fontSize: 12, fontWeight: 800, display: 'block', marginBottom: 4 }}>
                      Focus Sound (Optional)
                    </label>
                    <input
                      type="text"
                      value={formFocusPhoneme}
                      onChange={(e) => setFormFocusPhoneme(e.target.value)}
                      placeholder="e.g. /θ/ or Word Stress"
                      style={{
                        width: '100%',
                        padding: '8px 10px',
                        borderRadius: 10,
                        border: `1.5px solid ${c.border}`,
                        background: c.surface,
                        color: c.text,
                        fontSize: 13,
                      }}
                    />
                  </div>
                </div>

                <div>
                  <label style={{ fontSize: 12, fontWeight: 800, display: 'block', marginBottom: 4 }}>
                    Coaching Tip or Meaning (Optional)
                  </label>
                  <input
                    type="text"
                    value={formTip}
                    onChange={(e) => setFormTip(e.target.value)}
                    placeholder="e.g. Stress the second syllable and link smoothly"
                    style={{
                      width: '100%',
                      padding: '8px 10px',
                      borderRadius: 10,
                      border: `1.5px solid ${c.border}`,
                      background: c.surface,
                      color: c.text,
                      fontSize: 13,
                    }}
                  />
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 6 }}>
                  <ChunkyButton
                    tone="soft"
                    type="button"
                    onClick={() => setShowItemModal(false)}
                  >
                    Cancel
                  </ChunkyButton>
                  <ChunkyButton tone="navy" type="submit" disabled={savingItem}>
                    {savingItem ? 'Saving & Computing IPA...' : 'Save to Library'}
                  </ChunkyButton>
                </div>
              </form>
            </GameCard>
          </div>
        </div>
      )}

      {/* =====================================================================
          MODAL 3: SETTINGS & CONNECTION PREFERENCES
         ===================================================================== */}
      {showSettingsModal && (
        <div
          className="tv-modal-backdrop"
          onClick={() => setShowSettingsModal(false)}
        >
          <div
            className="tv-modal-scroll-box"
            onClick={(e) => e.stopPropagation()}
            style={{ maxWidth: 440 }}
          >
            <GameCard innerStyle={{ padding: '18px 20px' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 14,
                }}
              >
                <div style={{ fontSize: 18, fontWeight: 900 }}>⚙️ Coach Settings</div>
                <button
                  type="button"
                  onClick={() => setShowSettingsModal(false)}
                  style={{
                    border: 'none',
                    background: 'transparent',
                    color: c.muted,
                    cursor: 'pointer',
                  }}
                >
                  <X size={20} />
                </button>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontWeight: 800, fontSize: 14 }}>Theme Appearance</div>
                    <div style={{ fontSize: 12, color: c.muted }}>Switch Light or Dark studio theme</div>
                  </div>
                  <ChunkyButton tone="soft" size="sm" onClick={toggleTheme}>
                    {isDark ? <Sun size={15} /> : <Moon size={15} />}{' '}
                    {isDark ? 'Light Mode' : 'Dark Mode'}
                  </ChunkyButton>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontWeight: 800, fontSize: 14 }}>Auto-Submit When Done</div>
                    <div style={{ fontSize: 12, color: c.muted }}>
                      Automatically score ~1s after you stop speaking
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={autoStopVad}
                    onChange={(e) => setAutoStopVad(e.target.checked)}
                    style={{ width: 18, height: 18, cursor: 'pointer' }}
                  />
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontWeight: 800, fontSize: 14 }}>Audio Cues & Chimes</div>
                    <div style={{ fontSize: 12, color: c.muted }}>
                      Play gentle chimes on start, stop, and star rewards
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={soundEffectsOn}
                    onChange={(e) => setSoundEffectsOn(e.target.checked)}
                    style={{ width: 18, height: 18, cursor: 'pointer' }}
                  />
                </div>

                <div
                  style={{
                    paddingTop: 12,
                    borderTop: `1px solid ${c.border}`,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 800, fontSize: 14 }}>Speech Scoring Server</div>
                    <div style={{ fontSize: 11, color: c.muted }}>
                      {serverOnline ? '🟢 Connected' : '🟠 Offline'} ·{' '}
                      {usingServerDb ? 'SQLite Server DB Active' : 'Local Sync Active'}
                      {latencyMs ? ` · Last: ${latencyMs}ms` : ''}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <ChunkyButton
                      tone={apiTarget === 'aws' ? 'navy' : 'soft'}
                      size="sm"
                      onClick={() => setApiTarget('aws')}
                    >
                      ☁️ AWS
                    </ChunkyButton>
                    <ChunkyButton
                      tone={apiTarget === 'local' ? 'navy' : 'soft'}
                      size="sm"
                      onClick={() => setApiTarget('local')}
                    >
                      💻 Local
                    </ChunkyButton>
                  </div>
                </div>
              </div>
            </GameCard>
          </div>
        </div>
      )}

      {/* =====================================================================
          MODAL 4: MOBILE PRACTICE FILTER POPUP
         ===================================================================== */}
      {showFilterModal && (
        <div
          className="tv-modal-backdrop"
          onClick={() => setShowFilterModal(false)}
        >
          <div
            className="tv-modal-scroll-box"
            onClick={(e) => e.stopPropagation()}
            style={{ maxWidth: 420 }}
          >
            <GameCard innerStyle={{ padding: '18px 20px' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 14,
                }}
              >
                <div style={{ fontSize: 18, fontWeight: 900, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <SlidersHorizontal size={18} color="#0D3B84" /> Filter Practice Deck
                </div>
                <button
                  type="button"
                  onClick={() => setShowFilterModal(false)}
                  style={{
                    border: 'none',
                    background: 'transparent',
                    color: c.muted,
                    cursor: 'pointer',
                  }}
                >
                  <X size={20} />
                </button>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                {/* Language */}
                <div>
                  <label style={{ fontSize: 12, fontWeight: 800, display: 'block', marginBottom: 6 }}>
                    Language
                  </label>
                  <select
                    value={selectedLang}
                    onChange={(e) => {
                      const newLang = e.target.value;
                      setSelectedLang(newLang);
                      const firstForLang = libraryItems.find((i) => i.language === newLang);
                      if (firstForLang) selectLibraryItemForPractice(firstForLang);
                    }}
                    style={{
                      width: '100%',
                      padding: '9px 12px',
                      borderRadius: 10,
                      border: `1.5px solid ${c.border}`,
                      background: c.surface,
                      color: c.text,
                      fontWeight: 800,
                      fontSize: 13.5,
                    }}
                  >
                    {LANGUAGE_OPTIONS.map((l) => (
                      <option key={l.code} value={l.code}>
                        {l.flag} {l.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Practice Mode (Words vs Phrases) */}
                <div>
                  <label style={{ fontSize: 12, fontWeight: 800, display: 'block', marginBottom: 6 }}>
                    Practice Mode
                  </label>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
                    {(['all', 'word', 'phrase'] as const).map((t) => (
                      <button
                        key={t}
                        type="button"
                        onClick={() => setFilterType(t)}
                        style={{
                          padding: '8px 6px',
                          borderRadius: 10,
                          border: `1.5px solid ${filterType === t ? '#0D3B84' : c.border}`,
                          background:
                            filterType === t ? (isDark ? '#2F5FC4' : '#0D3B84') : c.surface,
                          color: filterType === t ? '#FFFFFF' : c.text,
                          fontWeight: 800,
                          fontSize: 12,
                          cursor: 'pointer',
                        }}
                      >
                        {t === 'all' ? 'All' : t === 'word' ? '🔤 Words' : '💬 Phrases'}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Difficulty Level */}
                <div>
                  <label style={{ fontSize: 12, fontWeight: 800, display: 'block', marginBottom: 6 }}>
                    Difficulty Level
                  </label>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                    {(['all', 'Beginner', 'Intermediate', 'Advanced'] as const).map((d) => (
                      <button
                        key={d}
                        type="button"
                        onClick={() => setFilterDifficulty(d)}
                        style={{
                          padding: '8px 10px',
                          borderRadius: 10,
                          border: `1.5px solid ${filterDifficulty === d ? '#F89A1C' : c.border}`,
                          background:
                            filterDifficulty === d ? 'rgba(248,154,28,0.18)' : c.surface,
                          color: filterDifficulty === d ? '#F89A1C' : c.text,
                          fontWeight: 800,
                          fontSize: 12,
                          cursor: 'pointer',
                        }}
                      >
                        {d === 'all' ? 'All Levels' : d}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Category */}
                <div>
                  <label style={{ fontSize: 12, fontWeight: 800, display: 'block', marginBottom: 6 }}>
                    Topic Category
                  </label>
                  <select
                    value={filterCategory}
                    onChange={(e) => setFilterCategory(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '9px 12px',
                      borderRadius: 10,
                      border: `1.5px solid ${c.border}`,
                      background: c.surface,
                      color: c.text,
                      fontWeight: 700,
                      fontSize: 13,
                    }}
                  >
                    <option value="all">All Categories</option>
                    {CATEGORY_OPTIONS.map((cat) => (
                      <option key={cat} value={cat}>
                        {cat}
                      </option>
                    ))}
                  </select>
                </div>

                <ChunkyButton
                  tone="navy"
                  fullWidth
                  onClick={() => setShowFilterModal(false)}
                >
                  Show {filteredItems.length} Matching Items
                </ChunkyButton>
              </div>
            </GameCard>
          </div>
        </div>
      )}
      </div>
    </>
  );
}

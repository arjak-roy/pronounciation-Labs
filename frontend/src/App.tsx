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

type MainTab = 'practice' | 'library' | 'progress';
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
  const [autoStopVad, setAutoStopVad] = useState<boolean>(true);
  const [soundEffectsOn, setSoundEffectsOn] = useState<boolean>(true);

  const apiBase = apiTarget === 'aws' ? '/aws-api' : '/local-api';

  // --- Library Database State ---
  const [libraryItems, setLibraryItems] = useState<LibraryItem[]>(() => loadFallbackLibrary());
  const [libraryStats, setLibraryStats] = useState<LibraryStats | null>(null);
  const [usingServerDb, setUsingServerDb] = useState<boolean>(false);

  // --- Practice Filters & Active Item ---
  const [selectedLang, setSelectedLang] = useState<string>('en');
  const [filterType, setFilterType] = useState<'all' | 'word' | 'phrase'>('all');
  const [filterDifficulty, setFilterDifficulty] = useState<string>('all');
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const [activeItemId, setActiveItemId] = useState<number>(9); // Default: "She can see the sea"
  const [referenceText, setReferenceText] = useState<string>('She can see the sea');
  const [speechSpeed, setSpeechSpeed] = useState<number>(1.0);

  // --- Add / Edit Word or Phrase Modal State ---
  const [showItemModal, setShowItemModal] = useState<boolean>(false);
  const [editingItem, setEditingItem] = useState<LibraryItem | null>(null);
  const [formText, setFormText] = useState<string>('');
  const [formType, setFormType] = useState<'word' | 'phrase'>('phrase');
  const [formLang, setFormLang] = useState<string>('en');
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
      } else {
        setServerOnline(false);
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
    if (recordedAudioUrl) URL.revokeObjectURL(recordedAudioUrl);
    setRecordedAudioUrl(URL.createObjectURL(file));
    await submitAudioForAssessment(file, file.name);
  };

  // --- Submit Audio to Mode B Pronunciation Engine ---
  const submitAudioForAssessment = async (blob: Blob, filename: string) => {
    const cleanRef = referenceText.trim();
    if (!cleanRef) return;

    setIsAnalyzing(true);
    setErrorMsg(null);
    if (avatarRuntimeRef.current) {
      avatarRuntimeRef.current.setActivity('thinking');
      avatarRuntimeRef.current.setHeadTurn(0);
    }
    setCoachBubble('🧠 Checking every sound, word stress beat, and smooth word link...');

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
    <div style={{ maxWidth: 1180, margin: '0 auto', padding: '14px 16px 48px' }}>
      {/* =====================================================================
          1. CLEAN LEARNER HEADER WITH 3 PRIMARY NAVIGATION TABS
         ===================================================================== */}
      <GameCard style={{ marginBottom: 16, padding: '12px 16px' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 12,
          }}
        >
          {/* Brand */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <IconTile tone="navy" size={44}>
              <Mic size={22} />
            </IconTile>
            <div>
              <div style={{ fontWeight: 900, fontSize: 18, letterSpacing: '-0.02em' }}>
                GTS Speak & Shine Coach
              </div>
              <div style={{ fontSize: 12, color: c.muted, display: 'flex', alignItems: 'center', gap: 6 }}>
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: 99,
                    backgroundColor: serverOnline ? '#16A34A' : '#F59E0B',
                    display: 'inline-block',
                  }}
                />
                <span>
                  {LANGUAGE_OPTIONS.find((l) => l.code === selectedLang)?.flag || '🌐'}{' '}
                  {LANGUAGE_OPTIONS.find((l) => l.code === selectedLang)?.label || 'All Languages'} · Interactive Pronunciation Studio
                </span>
              </div>
            </div>
          </div>

          {/* 3 Primary Learner Navigation Tabs */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <ChunkyButton
              tone={activeTab === 'practice' ? 'navy' : 'soft'}
              size="sm"
              onClick={() => setActiveTab('practice')}
            >
              <Target size={15} /> Practice Studio
            </ChunkyButton>
            <ChunkyButton
              tone={activeTab === 'library' ? 'navy' : 'soft'}
              size="sm"
              onClick={() => setActiveTab('library')}
            >
              <BookOpen size={15} /> Word & Phrase Bank ({libraryItems.length})
            </ChunkyButton>
            <ChunkyButton
              tone={activeTab === 'progress' ? 'navy' : 'soft'}
              size="sm"
              onClick={() => setActiveTab('progress')}
            >
              <TrendingUp size={15} /> My Progress
            </ChunkyButton>
          </div>

          {/* Right HUD: Streak, Stars & Settings Gear */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Pill tone="orange">
              <Flame size={13} /> {streak} Streak
            </Pill>
            <Pill tone="blue">
              <Award size={13} /> {masteredCount} Mastered
            </Pill>
            <ChunkyButton
              tone="soft"
              size="sm"
              onClick={() => setShowSettingsModal(true)}
              title="Audio & Connection Settings"
            >
              <Settings size={15} />
            </ChunkyButton>
          </div>
        </div>
      </GameCard>

      {/* =====================================================================
          TAB 1: PRACTICE STUDIO (USER-CENTRIC PRONUNCIATION WORKSPACE)
         ===================================================================== */}
      {activeTab === 'practice' && (
        <>
          {/* A. Simple Filter & Curriculum Strip */}
          <GameCard style={{ marginBottom: 16, padding: '12px 16px' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: 10,
                marginBottom: 12,
              }}
            >
              {/* Language & Mode Filters */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
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
            <div
              style={{
                display: 'flex',
                gap: 10,
                overflowX: 'auto',
                paddingBottom: 6,
              }}
            >
              {filteredItems.map((item) => {
                const isSelected = item.id === activeItemId;
                const itemStars = starsForScore(item.best_score || 0);
                return (
                  <div
                    key={item.id}
                    onClick={() => selectLibraryItemForPractice(item)}
                    style={{
                      minWidth: 215,
                      maxWidth: 240,
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
                      flexShrink: 0,
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
            <GameCard style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
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
                style={{
                  width: '100%',
                  height: 230,
                  borderRadius: 16,
                  overflow: 'hidden',
                  background: isDark
                    ? 'radial-gradient(circle at 50% 35%, #1e293b, #0f172a)'
                    : 'radial-gradient(circle at 50% 35%, #DBEAFE, #EFF6FF)',
                  border: `1.5px solid ${c.border}`,
                }}
              />

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
            </GameCard>

            {/* Right Column: Learner Practice Stage */}
            <GameCard style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', gap: 14 }}>
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
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
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
                  }}
                >
                  {speechSpeed === 1.0 ? '⚡ Normal 1.0x' : '🐢 Slow 0.75x'}
                </button>
              </div>

              {/* Interactive Word & Syllable Tiles */}
              <div
                style={{
                  padding: '18px 16px',
                  borderRadius: 16,
                  background: isDark ? '#0F172A' : '#F8FAFC',
                  border: `1.5px solid ${c.border}`,
                  display: 'flex',
                  flexWrap: 'wrap',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 10,
                  minHeight: 128,
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
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
                <ChunkyButton
                  tone="soft"
                  onClick={() => speakCoachPhrase()}
                  disabled={isRecording || isAnalyzing}
                >
                  <Volume2 size={17} /> 1. Listen to Coach
                </ChunkyButton>

                {!isRecording ? (
                  <ChunkyButton
                    tone="orange"
                    onClick={startRecording}
                    disabled={isAnalyzing}
                  >
                    <Mic size={17} /> {isAnalyzing ? 'Scoring Your Voice...' : '2. Speak Now'}
                  </ChunkyButton>
                ) : (
                  <ChunkyButton tone="red" onClick={stopRecordingAndScore}>
                    <Square size={17} /> Stop & Score
                  </ChunkyButton>
                )}

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

                <label style={{ cursor: 'pointer', marginLeft: 'auto' }} title="Upload a WAV file">
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
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
                  gap: 10,
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
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(310px, 1fr))',
              gap: 12,
            }}
          >
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
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: 12,
            }}
          >
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
          onClick={() => setShowResultPopup(false)}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 90,
            background: 'rgba(15, 23, 42, 0.72)',
            backdropFilter: 'blur(6px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 16,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{ width: '100%', maxWidth: 500 }}
          >
            <GameCard style={{ padding: '22px 24px', textAlign: 'center' }}>
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

              <div style={{ fontSize: 22, fontWeight: 900, marginBottom: 4 }}>
                {assessResult.overall_score >= 90
                  ? '🌟 Outstanding Pronunciation!'
                  : assessResult.overall_score >= 75
                  ? '👏 Great Job — Very Clear!'
                  : '💪 Good Effort — Let’s Polish It!'}
              </div>
              <div style={{ fontSize: 13, color: c.muted, marginBottom: 16 }}>
                “{referenceText}”
              </div>

              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 18 }}>
                <ProgressRing
                  value={assessResult.overall_score}
                  size={112}
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

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: 10,
                  marginBottom: 18,
                }}
              >
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

              <div
                style={{
                  display: 'flex',
                  gap: 10,
                  justifyContent: 'center',
                  flexWrap: 'wrap',
                }}
              >
                {recordedAudioUrl && (
                  <ChunkyButton tone="green" onClick={playMyVoice}>
                    <Play size={16} /> Hear My Voice
                  </ChunkyButton>
                )}
                <ChunkyButton
                  tone="orange"
                  onClick={() => {
                    setShowResultPopup(false);
                    startRecording();
                  }}
                >
                  <Mic size={16} /> Try Again
                </ChunkyButton>
                <ChunkyButton tone="navy" onClick={() => setShowResultPopup(false)}>
                  <CheckCircle2 size={16} /> See Sound Tips
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
          onClick={() => setShowItemModal(false)}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 95,
            background: 'rgba(15, 23, 42, 0.72)',
            backdropFilter: 'blur(6px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 16,
          }}
        >
          <div onClick={(e) => e.stopPropagation()} style={{ width: '100%', maxWidth: 520 }}>
            <GameCard style={{ padding: '20px 22px' }}>
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

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
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

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
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
          onClick={() => setShowSettingsModal(false)}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 95,
            background: 'rgba(15, 23, 42, 0.72)',
            backdropFilter: 'blur(6px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 16,
          }}
        >
          <div onClick={(e) => e.stopPropagation()} style={{ width: '100%', maxWidth: 440 }}>
            <GameCard style={{ padding: '20px 22px' }}>
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
    </div>
  );
}

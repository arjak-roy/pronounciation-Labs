import React, { useEffect, useRef, useState } from 'react';
import {
  Award,
  Flame,
  Globe,
  Mic,
  Play,
  RefreshCw,
  Server,
  Sparkles,
  Square,
  Sun,
  Moon,
  Upload,
  Volume2,
  Zap,
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
  status: 'smooth_link' | 'natural_pause' | 'choppy_pause';
  coaching_tip?: string | null;
}

interface ConnectedSpeechReport {
  linking_score: number;
  smooth_links_count: number;
  choppy_Pause_count: number;
  intonation_contour_hz: number[];
  intonation_type: 'falling_statement' | 'rising_question' | 'flat_monotone' | 'expressive';
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

const QUESTS = [
  {
    id: 'q1',
    stage: 'STAGE 1',
    title: 'Sibilant Warmup',
    phrase: 'She can see the sea',
    focus: '/ʃ/ vs /s/ & long /iː/',
    tone: 'navy' as const,
  },
  {
    id: 'q2',
    stage: 'STAGE 2',
    title: 'Dental Fricative TH',
    phrase: 'She thought about the pronunciation',
    focus: 'Unvoiced /θ/ & voiced /ð/',
    tone: 'orange' as const,
  },
  {
    id: 'q3',
    stage: 'STAGE 3',
    title: 'Labial /w/ vs /v/',
    phrase: 'The weather is very warm today',
    focus: 'Rounded lips /w/ vs teeth /v/',
    tone: 'teal' as const,
  },
  {
    id: 'q4',
    stage: 'STAGE 4',
    title: 'Consonant Clusters',
    phrase: 'Three free throws in a row',
    focus: '/θɹ/ vs /fɹ/ articulation',
    tone: 'purple' as const,
  },
  {
    id: 'q5',
    stage: 'BOSS STAGE',
    title: 'Plosive Speed Drill',
    phrase: 'Peter Piper picked a peck of pickled peppers',
    focus: 'Aspirated /p/ & rhythm',
    tone: 'rose' as const,
  },
];

const PHONEME_TIPS: Record<string, string> = {
  θ: "Place the tip of your tongue gently between your front teeth and blow air softly without vibrating your throat (as in 'think').",
  ð: "Place your tongue tip gently between your teeth and vibrate your vocal cords (as in 'the' or 'this').",
  ʃ: "Round your lips slightly and push air softly over the blade of your tongue for a 'shh' sound (as in 'she').",
  s: "Keep your teeth close together and direct a crisp stream of air over the tip of your tongue.",
  ɹ: "Curl your tongue tip slightly back without touching the roof of your mouth.",
  'iː': "Smile slightly and hold the 'eee' vowel sound slightly longer (as in 'see').",
  ɪ: "Keep your lips and tongue relaxed for a short, crisp 'ih' sound (as in 'it').",
  æ: "Drop your jaw wider with your tongue low in the front of your mouth (as in 'can').",
  v: "Touch your upper teeth lightly to your bottom lip and buzz your voice.",
  w: "Round your lips into a tight circle without touching your teeth to your lip.",
};

function ipaToViseme(ipa?: string | null): string {
  if (!ipa) return 'neutral';
  const s = ipa.toLowerCase();
  if (s.includes('f') || s.includes('v') || s.includes('θ') || s.includes('ð')) return 'fv';
  if (s.includes('ɔ') || s.includes('o') || s.includes('u') || s.includes('ʊ') || s.includes('w')) return 'o';
  if (s.includes('i') || s.includes('ɪ') || s.includes('e') || s.includes('s') || s.includes('z') || s.includes('ʃ')) {
    return 'wide';
  }
  if (s.includes('ɑ') || s.includes('æ') || s.includes('ʌ') || s.includes('ɐ') || s.includes('ə') || s.includes('a')) {
    return 'open';
  }
  return 'neutral';
}

export default function App() {
  const { isDark, toggleTheme } = useTheme();
  const card = useCardColors();

  // Switchable API Base URL: AWS Live Server (via Vite CORS-free proxy) vs Local Backend
  const [apiBaseUrl, setApiBaseUrl] = useState<string>('/aws-api');
  const [engineReady, setEngineReady] = useState<boolean>(false);
  const [engineProvider, setEngineProvider] = useState<string>('Checking API...');

  // Quest & Phrase State
  const [activeQuestId, setActiveQuestId] = useState<string>('q1');
  const [referenceText, setReferenceText] = useState<string>('She can see the sea');
  const [language, setLanguage] = useState<string>('en');
  const [speed, setSpeed] = useState<number>(1.0);

  // Word & Assessment State
  const [previewWords, setPreviewWords] = useState<WordItem[]>([]);
  const [assessment, setAssessment] = useState<AssessResponse | null>(null);
  const [showResultPopup, setShowResultPopup] = useState<boolean>(false);
  const [selectedWordIdx, setSelectedWordIdx] = useState<number>(0);
  const [speakingWordIdx, setSpeakingWordIdx] = useState<number | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);

  // Gamification State
  const [xp, setXp] = useState<number>(120);
  const [streak, setStreak] = useState<number>(2);
  const [questStars, setQuestStars] = useState<Record<string, number>>({ q1: 3 });

  // Recording & Loading State
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [countdownNum, setCountdownNum] = useState<number | null>(null);
  const [autoSubmitVad, setAutoSubmitVad] = useState<boolean>(true);
  const [micLevel, setMicLevel] = useState<number>(0);
  const [isPlayingUserVoice, setIsPlayingUserVoice] = useState<boolean>(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisStep, setAnalysisStep] = useState<number>(1);
  const [audioBlobUrl, setAudioBlobUrl] = useState<string | null>(null);
  const [characterId, setCharacterId] = useState<string>('girl-3d');
  const [currentViseme, setCurrentViseme] = useState<string>('neutral');
  const [coachBubble, setCoachBubble] = useState<string>(
    'Tap any word card to see how my mouth shapes each sound, or press RECORD & SCORE!'
  );

  // Refs
  const confettiCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const avatarHostRef = useRef<HTMLDivElement | null>(null);
  const avatarRuntimeRef = useRef<any>(null);
  const audioSamplesRef = useRef<Float32Array | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const recordedChunksRef = useRef<Float32Array[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const isRecordingRef = useRef<boolean>(false);
  const autoSubmitVadRef = useRef<boolean>(true);
  const userAudioPlayerRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    autoSubmitVadRef.current = autoSubmitVad;
  }, [autoSubmitVad]);

  // 1. Check API Health
  useEffect(() => {
    let cancelled = false;
    async function checkApi() {
      try {
        const res = await fetch(`${apiBaseUrl}/health`);
        const data = await res.json();
        if (!cancelled && data.mode_b?.ready) {
          setEngineReady(true);
          setEngineProvider(`${data.mode_b.device.toUpperCase()} · ${data.mode_b.provider}`);
        }
      } catch {
        if (!cancelled) {
          setEngineReady(false);
          setEngineProvider('Unreachable (Check CORS or Local Server)');
        }
      }
    }
    checkApi();
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl]);

  // 2. Fetch Expected IPA Preview whenever phrase/language changes
  useEffect(() => {
    const timer = setTimeout(async () => {
      if (!referenceText.trim()) {
        setPreviewWords([]);
        return;
      }
      try {
        const res = await fetch(
          `${apiBaseUrl}/api/v1/pronunciation/phonemize?text=${encodeURIComponent(
            referenceText
          )}&language=${encodeURIComponent(language)}`
        );
        if (res.ok) {
          const data = await res.json();
          setPreviewWords(data.words || []);
          return;
        }
      } catch {
        // Fallback word tokens
      }
      setPreviewWords(
        referenceText
          .trim()
          .split(/\s+/)
          .map((w) => ({ word: w, expected_ipa: '...', phonemes: [] }))
      );
    }, 200);
    return () => clearTimeout(timer);
  }, [referenceText, language, apiBaseUrl]);

  // 3. Initialize PixiJS 3D Avatar Coach
  useEffect(() => {
    let mounted = true;
    async function loadAvatar() {
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
        if (!mounted || !avatarHostRef.current) return;
        const canvas = renderer.getSurfaceElement();
        if (canvas) avatarHostRef.current.appendChild(canvas);
        await rt.load(`/characters/${characterId}`);
        rt.on('mouth.changed', ({ mouth }: { mouth: string }) => {
          setCurrentViseme(mouth);
        });
        avatarRuntimeRef.current = rt;
      } catch (e) {
        console.warn('Avatar load warning:', e);
      }
    }
    loadAvatar();
    return () => {
      mounted = false;
    };
  }, [characterId]);

  const displayedWords: WordItem[] = assessment?.words || previewWords;
  const selectedWord: WordItem | undefined = displayedWords[selectedWordIdx];

  function animateWordVisemes(wordObj?: WordItem, durationMs = 420) {
    const rt = avatarRuntimeRef.current;
    if (!rt || !wordObj) return;
    const list = (wordObj.phonemes || []).map((p) =>
      typeof p === 'string' ? p : p.expected || p.heard || ''
    ).filter(Boolean);
    if (list.length === 0) return;

    const step = Math.max(70, durationMs / list.length);
    list.forEach((ph, idx) => {
      setTimeout(() => {
        const v = ipaToViseme(ph);
        rt.setMouth(v);
        setCurrentViseme(`${v} (/${ph}/)`);
      }, idx * step);
    });
    setTimeout(() => {
      rt.setMouth('neutral');
      setCurrentViseme('neutral');
    }, list.length * step + 70);
  }

  function speakPhrase(onComplete?: () => void) {
    if (!referenceText.trim()) return;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(referenceText);
    utter.lang = language;
    utter.rate = speed;

    const rt = avatarRuntimeRef.current;
    const words = displayedWords;
    const approxMs = Math.max(1100, (words.length * 360) / speed);
    const stepMs = approxMs / Math.max(1, words.length);
    const timers: number[] = [];

    utter.onstart = () => {
      if (rt) {
        rt.setActivity('speaking');
        rt.setExpression('happy', 0.6);
      }
      setCoachBubble(`🔊 Listen closely and watch my mouth shapes for "${referenceText}"`);
      words.forEach((w, i) => {
        timers.push(
          window.setTimeout(() => {
            setSpeakingWordIdx(i);
            animateWordVisemes(w, stepMs * 0.85);
          }, i * stepMs)
        );
      });
    };

    utter.onend = () => {
      timers.forEach(clearTimeout);
      setSpeakingWordIdx(null);
      if (rt) {
        rt.setActivity('idle');
        rt.setMouth('neutral');
      }
      setCoachBubble('🎙️ Now your turn! Press RECORD & SCORE and repeat the phrase.');
      if (onComplete) onComplete();
    };

    window.speechSynthesis.speak(utter);
  }

  function startGuidedShadowQuest() {
    if (isRecordingRef.current || countdownNum !== null) return;
    speakPhrase(() => {
      let count = 3;
      setCountdownNum(count);
      playTone(440, 0.08);

      const iv = window.setInterval(() => {
        count -= 1;
        if (count > 0) {
          setCountdownNum(count);
          playTone(440, 0.08);
        } else {
          clearInterval(iv);
          setCountdownNum(null);
          playTone(880, 0.12);
          if (!isRecordingRef.current) void toggleRecording(true);
        }
      }, 620);
    });
  }

  async function stopRecordingAndScore() {
    if (!isRecordingRef.current) return;
    isRecordingRef.current = false;
    setIsRecording(false);
    setMicLevel(0);
    playTone(480, 0.09);

    const inRate = audioCtxRef.current?.sampleRate || 48000;
    processorRef.current?.disconnect();
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    try {
      await audioCtxRef.current?.close();
    } catch {
      // ignore if already closed
    }

    const merged = mergeBuffers(recordedChunksRef.current);
    const ds16k = downsampleBuffer(merged, inRate, 16000);
    const trimmed16k = trimSilenceClientSide(ds16k, 16000);
    audioSamplesRef.current = trimmed16k;

    const wavBlob = encodeMono16kWav(trimmed16k, 16000);
    setAudioBlobUrl(URL.createObjectURL(wavBlob));
    await runAssessment(wavBlob, 'shadowing_16k.wav');
  }

  async function toggleRecording(skipStartBeep = false) {
    const rt = avatarRuntimeRef.current;
    if (!isRecordingRef.current) {
      try {
        if (!skipStartBeep) playTone(660, 0.09);
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
        const ctx = new AudioContext();
        const inRate = ctx.sampleRate || 48000;
        const source = ctx.createMediaStreamSource(stream);
        const proc = ctx.createScriptProcessor(2048, 1, 1);
        recordedChunksRef.current = [];

        let speechDetected = false;
        let silentFramesAfterSpeech = 0;
        const framesFor950ms = Math.ceil((0.95 * inRate) / 2048);

        proc.onaudioprocess = (e) => {
          if (!isRecordingRef.current) return;
          const ch = e.inputBuffer.getChannelData(0);
          recordedChunksRef.current.push(new Float32Array(ch));

          let sumSq = 0;
          for (let i = 0; i < ch.length; i++) sumSq += ch[i] * ch[i];
          const rms = Math.sqrt(sumSq / ch.length);
          setMicLevel(Math.min(100, Math.round(rms * 380)));

          if (rms > 0.022) {
            speechDetected = true;
            silentFramesAfterSpeech = 0;
          } else if (speechDetected) {
            silentFramesAfterSpeech++;
            if (autoSubmitVadRef.current && silentFramesAfterSpeech >= framesFor950ms) {
              void stopRecordingAndScore();
            }
          }
        };

        source.connect(proc);
        proc.connect(ctx.destination);

        mediaStreamRef.current = stream;
        audioCtxRef.current = ctx;
        processorRef.current = proc;
        isRecordingRef.current = true;
        setIsRecording(true);

        if (rt) {
          rt.setActivity('listening');
          rt.setHeadTurn(0.12);
        }
        setCoachBubble(
          autoSubmitVadRef.current
            ? '👂 Listening! Speak now — I will auto-submit ~1s after you finish speaking.'
            : '👂 Listening! Speak each word clearly, then press STOP & SCORE.'
        );
      } catch (err: any) {
        alert('Microphone error: ' + err.message);
      }
    } else {
      await stopRecordingAndScore();
    }
  }

  function playMyVoiceSynced() {
    if (!audioBlobUrl) return;
    if (userAudioPlayerRef.current) {
      userAudioPlayerRef.current.pause();
      userAudioPlayerRef.current = null;
    }

    const audio = new Audio(audioBlobUrl);
    userAudioPlayerRef.current = audio;
    setIsPlayingUserVoice(true);

    const rt = avatarRuntimeRef.current;
    if (rt) rt.setActivity('speaking');
    setCoachBubble('▶️ Replaying your recorded voice with word-by-word karaoke sync...');

    let lastHighlighted = -1;
    const iv = window.setInterval(() => {
      if (audio.paused || audio.ended) {
        clearInterval(iv);
        setSpeakingWordIdx(null);
        setIsPlayingUserVoice(false);
        if (rt) {
          rt.setActivity('idle');
          rt.setMouth('neutral');
        }
        return;
      }
      const curMs = audio.currentTime * 1000;
      const words = assessment?.words || [];
      let activeIdx: number | null = null;
      words.forEach((w, idx) => {
        if (w.start_ms !== undefined && w.end_ms !== undefined) {
          if (curMs >= w.start_ms && curMs <= w.end_ms + 70) {
            activeIdx = idx;
          }
        }
      });
      setSpeakingWordIdx(activeIdx);
      if (activeIdx !== null && activeIdx !== lastHighlighted) {
        lastHighlighted = activeIdx;
        const wObj = words[activeIdx];
        if (wObj) animateWordVisemes(wObj, Math.max(180, (wObj.end_ms ?? 300) - (wObj.start_ms ?? 0)));
      }
    }, 30);

    audio.onended = () => {
      clearInterval(iv);
      setSpeakingWordIdx(null);
      setIsPlayingUserVoice(false);
      if (rt) {
        rt.setActivity('idle');
        rt.setMouth('neutral');
      }
    };

    void audio.play();
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setAudioBlobUrl(URL.createObjectURL(file));
    try {
      const ab = await file.arrayBuffer();
      const tmp = new AudioContext({ sampleRate: 16000 });
      const dec = await tmp.decodeAudioData(ab);
      audioSamplesRef.current = new Float32Array(dec.getChannelData(0));
      await tmp.close();
    } catch {
      audioSamplesRef.current = null;
    }
    await runAssessment(file, file.name);
  }

  async function runAssessment(blob: Blob, filename: string) {
    const rt = avatarRuntimeRef.current;
    setIsAnalyzing(true);
    setAnalysisStep(1);
    if (rt) {
      rt.setActivity('thinking');
      rt.setGaze('up');
    }
    setCoachBubble('🧠 Analyzing your 16kHz audio with Single-Pass Wav2Vec2 ONNX...');

    const s2Timer = window.setTimeout(() => setAnalysisStep(2), 110);
    const s3Timer = window.setTimeout(() => setAnalysisStep(3), 240);
    const t0 = performance.now();

    try {
      const fd = new FormData();
      fd.append('audio', blob, filename);
      fd.append('reference_text', referenceText);
      fd.append('language', language);

      const res = await fetch(`${apiBaseUrl}/api/v1/pronunciation/assess`, {
        method: 'POST',
        body: fd,
      });
      const elapsed = Math.round(performance.now() - t0);
      const data: AssessResponse = await res.json();

      if (!res.ok) {
        alert((data as any).detail || 'Assessment failed');
        return;
      }

      setLatencyMs(elapsed);
      setAssessment(data);
      setShowResultPopup(true);

      // Select lowest-scoring word automatically for coaching
      if (data.words?.length) {
        let minIdx = 0;
        data.words.forEach((w, idx) => {
          if ((w.score ?? 100) < (data.words[minIdx].score ?? 100)) minIdx = idx;
        });
        setSelectedWordIdx(minIdx);
      }

      const stars = data.overall_score >= 90 ? 3 : data.overall_score >= 75 ? 2 : 1;
      if (stars >= 2) {
        playVictoryChime(stars === 3);
        if (stars === 3) launchConfetti(confettiCanvasRef.current);
      } else {
        playTone(320, 0.12);
      }
      setXp((prev) => prev + stars * 20);
      setStreak((prev) => (data.overall_score >= 80 ? prev + 1 : 0));
      if (activeQuestId) {
        setQuestStars((prev) => ({
          ...prev,
          [activeQuestId]: Math.max(prev[activeQuestId] || 0, stars),
        }));
      }

      if (rt) {
        rt.setActivity('idle');
        rt.setGaze('forward');
        rt.setHeadTurn(0);
        if (data.overall_score >= 80) {
          rt.setExpression('happy', 1.0);
          rt.playGesture('nod');
          rt.setMouth('smile');
          setCoachBubble(`🎉 Great job! You scored ${data.overall_score}% in ${elapsed}ms!`);
        } else {
          rt.setExpression('concerned', 0.85);
          rt.playGesture('shake-head');
          setCoachBubble(`💡 Almost! Tap the highlighted word card to practice the sound.`);
        }
      }
    } catch (err: any) {
      alert(
        `Could not reach ${apiBaseUrl}/api/v1/pronunciation/assess (${err.message}). Try switching API Target in the top bar.`
      );
    } finally {
      clearTimeout(s2Timer);
      clearTimeout(s3Timer);
      setIsAnalyzing(false);
    }
  }

  function playSelectedWordAudioSlice() {
    const samples = audioSamplesRef.current;
    if (!samples || !selectedWord || selectedWord.start_ms === undefined || selectedWord.end_ms === undefined) {
      return;
    }
    const s = Math.max(0, Math.floor((selectedWord.start_ms / 1000) * 16000) - 480);
    const e = Math.min(samples.length, Math.ceil((selectedWord.end_ms / 1000) * 16000) + 640);
    const slice = samples.slice(s, Math.max(s + 320, e));
    const wav = encodeMono16kWav(slice, 16000);
    const audio = new Audio(URL.createObjectURL(wav));
    setSpeakingWordIdx(selectedWordIdx);
    animateWordVisemes(selectedWord, selectedWord.end_ms - selectedWord.start_ms);
    audio.onended = () => setSpeakingWordIdx(null);
    audio.play();
  }

  const totalStars = Object.values(questStars).reduce((a, b) => a + b, 0);

  return (
    <div className="gts-container">
      <canvas
        ref={confettiCanvasRef}
        style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 999 }}
      />
      {/* 1. GTS ACADEMY TOP HEADER BAR */}
      <GameCard
        style={{ marginBottom: 16 }}
        innerStyle={{
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 12,
          padding: '12px 16px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <IconTile tone="navy" size={44}>
            <Sparkles size={20} />
          </IconTile>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
              <span style={{ fontSize: 'clamp(0.95rem, 2.5vw, 1.12rem)', fontWeight: 900, letterSpacing: -0.3 }}>
                GTS ACADEMY · LANGUAGE LAB
              </span>
              <Pill label="MODE B ONNX" tone="orange" />
            </div>
            <div style={{ fontSize: 11, fontWeight: 600, color: card.muted, marginTop: 2 }}>
              <span
                style={{
                  display: 'inline-block',
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor: engineReady ? '#16A34A' : '#F59E0B',
                  marginRight: 6,
                }}
              />
              {engineProvider}
            </div>
          </div>
        </div>

        {/* Right: API Target Switcher + Stats + Theme Toggle */}
        <div className="gts-header-controls">
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Server size={14} color={card.muted} />
            <select
              value={apiBaseUrl}
              onChange={(e) => setApiBaseUrl(e.target.value)}
              style={{
                padding: '6px 8px',
                borderRadius: 10,
                border: `2px solid ${card.border}`,
                backgroundColor: card.surface,
                color: card.text,
                fontSize: 11,
                fontWeight: 800,
                cursor: 'pointer',
                maxWidth: 210,
              }}
            >
              <option value="/aws-api">☁️ AWS Live (18-234-54-50.nip.io)</option>
              <option value="/local-api">💻 Local Backend (127.0.0.1:8000)</option>
            </select>
          </div>

          <Pill label={`${streak} STREAK`} tone="orange" icon={<Flame size={12} />} />
          <Pill label={`${xp} XP`} tone="navy" icon={<Zap size={12} />} />
          <Pill label={`${totalStars}/15`} tone="amber" icon={<Award size={12} />} />

          <ChunkyButton
            size="sm"
            tone="secondary"
            label={isDark ? 'LIGHT' : 'DARK'}
            icon={isDark ? <Sun size={14} /> : <Moon size={14} />}
            onClick={toggleTheme}
          />
        </div>
      </GameCard>

      {/* 2. GTS ACADEMY STAGE QUEST CARDS (SWIPEABLE ON MOBILE, GRID ON DESKTOP) */}
      <div className="gts-quest-strip">
        {QUESTS.map((q) => {
          const isSelected = activeQuestId === q.id;
          return (
            <div key={q.id} className="gts-quest-card">
              <GameCard
                borderColor={isSelected ? '#F89A1C' : undefined}
                onClick={() => {
                  playTone(520, 0.06);
                  setActiveQuestId(q.id);
                  setReferenceText(q.phrase);
                  setAssessment(null);
                  setSelectedWordIdx(0);
                }}
                innerStyle={{
                  padding: 13,
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  gap: 6,
                  minHeight: 104,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Pill label={q.stage} tone={isSelected ? 'orange' : 'navy'} />
                  <StarRow earned={questStars[q.id] || 0} total={3} size={14} />
                </div>
                <div style={{ fontWeight: 900, fontSize: 13, color: card.text }}>"{q.phrase}"</div>
                <div style={{ fontSize: 11, fontWeight: 700, color: card.muted }}>
                  {q.focus}
                </div>
              </GameCard>
            </div>
          );
        })}
      </div>

      {/* 3. TARGET PHRASE & ACCENT BAR */}
      <GameCard
        style={{ marginBottom: 16 }}
        innerStyle={{ padding: 0 }}
      >
        <div className="gts-phrase-toolbar">
          <div className="gts-phrase-input-row">
            <span className="gts-phrase-pill-wrap">
              <Pill label="PHRASE" tone="navy" />
            </span>
            <input
              type="text"
              value={referenceText}
              onChange={(e) => {
                setActiveQuestId('');
                setReferenceText(e.target.value);
                setAssessment(null);
              }}
              placeholder="Type any sentence to practice shadowing..."
              style={{
                width: '100%',
                padding: '8px 12px',
                borderRadius: 12,
                border: `2px solid ${card.border}`,
                backgroundColor: card.surface,
                color: card.text,
                fontSize: 14,
                fontWeight: 800,
                outline: 'none',
              }}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              style={{
                width: '100%',
                padding: '8px 10px',
                borderRadius: 12,
                border: `2px solid ${card.border}`,
                backgroundColor: card.surface,
                color: card.text,
                fontSize: 12,
                fontWeight: 800,
              }}
            >
              <option value="en">🇺🇸 English (US)</option>
              <option value="en-gb">🇬🇧 English (UK)</option>
              <option value="es">🇪🇸 Spanish</option>
              <option value="fr">🇫🇷 French</option>
              <option value="de">🇩🇪 German</option>
              <option value="hi">🇮🇳 Hindi</option>
            </select>
          </div>

          <select
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
            style={{
              width: '100%',
              padding: '8px 10px',
              borderRadius: 12,
              border: `2px solid ${card.border}`,
              backgroundColor: card.surface,
              color: card.text,
              fontSize: 12,
              fontWeight: 800,
            }}
          >
            <option value={1.0}>1.0x Normal</option>
            <option value={0.85}>0.85x Practice</option>
            <option value={0.7}>0.70x Slow</option>
          </select>
        </div>
      </GameCard>

      {/* 4. MAIN SHADOWING WORKSPACE (3D AVATAR + TACTILE RAISED WORD TILES) */}
      <div className="gts-workspace-grid">
        {/* Left / Top on Mobile: GTS Avatar Coach Card */}
        <GameCard
          innerStyle={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            gap: 10,
            padding: 14,
          }}
        >
          <div
            style={{
              width: '100%',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <select
              value={characterId}
              onChange={(e) => setCharacterId(e.target.value)}
              style={{
                padding: '5px 8px',
                borderRadius: 8,
                border: `2px solid ${card.border}`,
                backgroundColor: card.surface,
                color: card.text,
                fontSize: 12,
                fontWeight: 800,
              }}
            >
              <option value="girl-3d">👩 Coach Maya</option>
              <option value="boy-3d">👦 Coach Leo</option>
              <option value="kawaii-demo">✨ Kawaii Coach</option>
              <option value="new-girl">👩‍🏫 Coach Aria</option>
              <option value="nano-banana">🍌 Nano Banana</option>
            </select>
            <Pill label={`MOUTH: ${currentViseme}`} tone="blue" />
          </div>

          <div className="gts-avatar-body">
            <div
              ref={avatarHostRef}
              className="gts-avatar-viewport"
              style={{
                backgroundColor: isDark ? '#0F172A' : '#EFF6FF',
              }}
            />

            <div
              style={{
                flex: 1,
                width: '100%',
                backgroundColor: card.surface,
                border: `2px solid ${card.border}`,
                borderRadius: 14,
                padding: '10px 12px',
                fontSize: 12,
                fontWeight: 700,
                color: card.text,
                textAlign: 'center',
              }}
            >
              {coachBubble}
            </div>
          </div>
        </GameCard>

        {/* Right: Interactive Phrase Stage + Chunky Action Buttons */}
        <GameCard
          style={{ position: 'relative' }}
          innerStyle={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            gap: 18,
            minHeight: 340,
          }}
        >
          {/* 3-2-1 Guided Countdown Overlay */}
          {countdownNum !== null && (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                backgroundColor: isDark ? 'rgba(15, 23, 42, 0.90)' : 'rgba(255, 255, 255, 0.90)',
                backdropFilter: 'blur(6px)',
                borderRadius: 22,
                zIndex: 35,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
              }}
            >
              <div style={{ fontSize: 72, fontWeight: 900, color: '#F89A1C', lineHeight: 1 }}>
                {countdownNum}
              </div>
              <div style={{ fontSize: 15, fontWeight: 900, color: card.text }}>
                GET READY TO SHADOW THE PHRASE ALOUD...
              </div>
            </div>
          )}

          {/* Analysis Loading Overlay */}
          {isAnalyzing && (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                backgroundColor: isDark ? 'rgba(15, 23, 42, 0.92)' : 'rgba(255, 255, 255, 0.92)',
                backdropFilter: 'blur(6px)',
                borderRadius: 22,
                zIndex: 30,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 12,
                padding: 24,
              }}
            >
              <RefreshCw size={42} color="#0D3B84" style={{ animation: 'spin 1s linear infinite' }} />
              <div style={{ fontSize: 18, fontWeight: 900 }}>Scoring Your Pronunciation...</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 13, fontWeight: 700 }}>
                <span style={{ color: analysisStep >= 1 ? '#16A34A' : card.muted }}>
                  ✓ 1. Normalizing 16kHz Mono WAV & Client Silence Gate
                </span>
                <span style={{ color: analysisStep >= 2 ? '#0D3B84' : card.muted }}>
                  ⚡ 2. Single-Pass Wav2Vec2 INT8 ONNX Inference (Logits + Layer 12)
                </span>
                <span style={{ color: analysisStep >= 3 ? '#F89A1C' : card.muted }}>
                  🔬 3. Forced Alignment & PanPhon Articulatory Scoring
                </span>
              </div>
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
            <Pill label="TAP ANY WORD TILE TO HEAR & INSPECT IPA" tone="navy" />
            <div style={{ display: 'flex', gap: 6 }}>
              <Pill label=">85% CORRECT" tone="green" />
              <Pill label="60-85% ALMOST" tone="amber" />
              <Pill label="<60% FIX SOUND" tone="red" />
            </div>
          </div>

          {/* Tactile GTS Raised Word Tiles */}
          <div
            className="gts-word-tiles-box"
            style={{
              backgroundColor: card.surface,
              border: `2px solid ${card.border}`,
            }}
          >
            {displayedWords.map((w, idx) => {
              const isSelected = selectedWordIdx === idx;
              const isSpeaking = speakingWordIdx === idx;
              const tone =
                w.status === 'correct'
                  ? getGameTone('green', isDark)
                  : w.status === 'warning'
                  ? getGameTone('orange', isDark)
                  : w.status === 'mispronounced' || w.status === 'omitted'
                  ? getGameTone('rose', isDark)
                  : getGameTone('white', isDark);

              const connReport =
                assessment?.connected_speech ||
                (assessment?.words ? computeClientConnectedSpeech(assessment.words) : null);
              const linkToNext = connReport?.links?.[idx];

              return (
                <React.Fragment key={idx}>
                  <Raised
                    face={tone.face}
                    shelf={tone.shelf}
                    border={isSpeaking || isSelected ? '#F89A1C' : card.border}
                    radius={16}
                    depth={isSpeaking ? 7 : 4}
                    onClick={() => {
                      setSelectedWordIdx(idx);
                      animateWordVisemes(w, 480);
                      const u = new SpeechSynthesisUtterance(w.word);
                      u.lang = language;
                      u.rate = speed;
                      window.speechSynthesis.cancel();
                      window.speechSynthesis.speak(u);
                    }}
                    innerStyle={{
                      padding: '10px 14px',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: 3,
                      minWidth: 78,
                    }}
                  >
                    {typeof w.score === 'number' && (
                      <span style={{ fontSize: 11, fontWeight: 900, color: tone.fg, opacity: 0.9 }}>
                        {w.score}%
                      </span>
                    )}
                    <span className="gts-word-surface" style={{ color: tone.fg }}>
                      {w.word}
                    </span>
                    <span
                      style={{
                        fontSize: 12,
                        fontWeight: 700,
                        color: tone.fg,
                        opacity: 0.85,
                        fontFamily: "'Lucida Sans Unicode', 'DejaVu Sans', monospace",
                      }}
                    >
                      /{w.expected_ipa}/
                    </span>
                    {w.status && w.status !== 'correct' && (
                      <span
                        style={{
                          fontSize: 10,
                          fontWeight: 800,
                          backgroundColor: 'rgba(0,0,0,0.18)',
                          color: tone.fg,
                          padding: '1px 6px',
                          borderRadius: 999,
                        }}
                      >
                        heard: /{w.heard_ipa || '∅'}/
                      </span>
                    )}
                  </Raised>

                  {/* Inline Connected-Speech Liaison / Pause Badge between adjacent words */}
                  {idx < displayedWords.length - 1 && (
                    <div
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        margin: '0 -4px',
                        alignSelf: 'center',
                      }}
                      title={
                        linkToNext
                          ? `${linkToNext.from_word}‿${linkToNext.to_word}: ${linkToNext.gap_ms}ms (${linkToNext.status})`
                          : 'Connected speech boundary'
                      }
                    >
                      <span
                        style={{
                          fontSize: 16,
                          fontWeight: 900,
                          color: !linkToNext
                            ? card.muted
                            : linkToNext.status === 'smooth_link'
                            ? '#16A34A'
                            : linkToNext.status === 'natural_pause'
                            ? '#F89A1C'
                            : '#DC2626',
                        }}
                      >
                        ‿
                      </span>
                      {linkToNext && (
                        <span
                          style={{
                            fontSize: 9,
                            fontWeight: 800,
                            color:
                              linkToNext.status === 'smooth_link'
                                ? '#16A34A'
                                : linkToNext.status === 'natural_pause'
                                ? '#F89A1C'
                                : '#DC2626',
                          }}
                        >
                          {linkToNext.gap_ms}ms
                        </span>
                      )}
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>

          {/* Live Mic Activity & Auto-Submit on Silence Bar */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: 10,
              backgroundColor: card.surface,
              border: `2px solid ${card.border}`,
              borderRadius: 14,
              padding: '8px 14px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: 1, minWidth: 140 }}>
              <span style={{ fontSize: 11, fontWeight: 900, color: card.muted, whiteSpace: 'nowrap' }}>
                MIC LEVEL
              </span>
              <div
                style={{
                  flex: 1,
                  height: 10,
                  backgroundColor: card.border,
                  borderRadius: 999,
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    width: `${micLevel}%`,
                    height: '100%',
                    backgroundColor: micLevel > 15 ? '#16A34A' : '#F89A1C',
                    transition: 'width 0.08s ease',
                  }}
                />
              </div>
            </div>

            <label
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                fontSize: 12,
                fontWeight: 800,
                color: card.text,
                cursor: 'pointer',
                userSelect: 'none',
              }}
            >
              <input
                type="checkbox"
                checked={autoSubmitVad}
                onChange={(e) => setAutoSubmitVad(e.target.checked)}
                style={{ width: 16, height: 16, accentColor: '#0D3B84', cursor: 'pointer' }}
              />
              Auto-Submit on Silence
            </label>
          </div>

          {/* GTS ChunkyButton Action Dock */}
          <div className="gts-action-dock">
            <div className="gts-primary-actions">
              <ChunkyButton
                tone="primary"
                label="1. Listen Coach"
                icon={<Volume2 size={17} />}
                onClick={() => speakPhrase()}
              />
              <ChunkyButton
                tone="accent"
                label="Guided Shadowing"
                icon={<Sparkles size={17} />}
                onClick={startGuidedShadowQuest}
              />
              <ChunkyButton
                tone={isRecording ? 'danger' : 'green'}
                label={isRecording ? 'Stop & Score' : '2. Record & Score'}
                icon={isRecording ? <Square size={17} /> : <Mic size={17} />}
                onClick={() => void toggleRecording(false)}
              />
              {audioBlobUrl && (
                <ChunkyButton
                  tone="secondary"
                  label={isPlayingUserVoice ? 'Playing Voice...' : 'Play My Voice'}
                  icon={<Play size={17} />}
                  onClick={playMyVoiceSynced}
                />
              )}
            </div>

            <div className="gts-secondary-actions">
              <ChunkyButton
                tone="secondary"
                size="sm"
                label="Upload WAV"
                icon={<Upload size={15} />}
                onClick={() => fileInputRef.current?.click()}
              />
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/wav,audio/*"
                style={{ display: 'none' }}
                onChange={handleFileUpload}
              />
            </div>
          </div>
        </GameCard>
      </div>

      {/* 5. SCORE METRICS & PHONEME DIAGNOSTICS (GTS STAT TILES + PROGRESS RING) */}
      {assessment && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
              gap: 12,
            }}
          >
            <GameCard innerStyle={{ padding: 14, display: 'flex', alignItems: 'center', gap: 14 }}>
              <ProgressRing
                size={64}
                stroke={7}
                percent={assessment.overall_score}
                color={assessment.overall_score >= 85 ? '#16A34A' : '#F89A1C'}
                trackColor={card.border}
              >
                <span style={{ fontSize: 16, fontWeight: 900 }}>{assessment.overall_score}%</span>
              </ProgressRing>
              <div>
                <div style={{ fontSize: 16, fontWeight: 900 }}>OVERALL SCORE</div>
                <div style={{ fontSize: 12, fontWeight: 700, color: card.muted }}>
                  {latencyMs ? `Scored in ${latencyMs}ms` : 'Mode B Assessment'}
                </div>
              </div>
            </GameCard>

            <StatTile
              tone="navy"
              icon={<Award size={20} />}
              value={`${assessment.metrics.accuracy}%`}
              label="PHONEME ACCURACY"
            />
            <StatTile
              tone="orange"
              icon={<Zap size={20} />}
              value={`${assessment.metrics.fluency}%`}
              label="FLUENCY & RHYTHM"
            />
            <StatTile
              tone="teal"
              icon={<Volume2 size={20} />}
              value={`${assessment.metrics.prosody_similarity}%`}
              label="PROSODY MATCH"
            />
          </div>

          {/* Selected Word Phoneme Breakdown Card */}
          {selectedWord && (
            <GameCard>
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
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ fontSize: 18, fontWeight: 900 }}>
                    WORD COACH: "{selectedWord.word.toUpperCase()}"
                  </span>
                  <Pill
                    label={`EXPECTED /${selectedWord.expected_ipa}/ · HEARD /${selectedWord.heard_ipa || selectedWord.expected_ipa}/`}
                    tone="navy"
                  />
                </div>
                {audioSamplesRef.current && selectedWord.start_ms !== undefined && (
                  <ChunkyButton
                    size="sm"
                    tone="secondary"
                    label="Play My Word Slice"
                    icon={<Play size={14} />}
                    onClick={playSelectedWordAudioSlice}
                  />
                )}
              </div>

              {/* MULTI-SYLLABLE WORD STRESS BREAKDOWN */}
              {(() => {
                const stressInfo =
                  selectedWord.word_stress ||
                  computeClientWordStress(
                    selectedWord.expected_ipa,
                    selectedWord.phonemes,
                    selectedWord.start_ms ?? 0,
                    selectedWord.end_ms ?? 240
                  );
                if (stressInfo.syllable_count <= 1) return null;
                return (
                  <div
                    style={{
                      backgroundColor: card.surface,
                      border: `2px solid ${card.border}`,
                      borderRadius: 16,
                      padding: '12px 14px',
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
                        marginBottom: 10,
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <Pill label={`MULTI-SYLLABLE STRESS (${stressInfo.syllable_count} SYLLABLES)`} tone="purple" />
                        <span
                          style={{
                            fontSize: 13,
                            fontWeight: 900,
                            fontFamily: "'Lucida Sans Unicode', 'DejaVu Sans', monospace",
                          }}
                        >
                          /{stressInfo.stress_pattern_display}/
                        </span>
                      </div>
                      <Pill
                        label={
                          stressInfo.stress_correct
                            ? '✓ PRIMARY STRESS ACCURATE'
                            : '⚠ SHIFT STRESS TO ˈPRIMARY'
                        }
                        tone={stressInfo.stress_correct ? 'green' : 'amber'}
                      />
                    </div>

                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: 8 }}>
                      {stressInfo.syllables.map((syl, sIdx) => {
                        const isPrimary = syl.stress === 'primary';
                        const sylTone = isPrimary
                          ? getGameTone('orange', isDark)
                          : getGameTone('soft', isDark);
                        return (
                          <Raised
                            key={sIdx}
                            face={sylTone.face}
                            shelf={sylTone.shelf}
                            radius={12}
                            depth={isPrimary ? 5 : 2}
                            innerStyle={{
                              padding: '8px 14px',
                              display: 'flex',
                              flexDirection: 'column',
                              alignItems: 'center',
                              gap: 2,
                            }}
                          >
                            <span style={{ fontSize: 10, fontWeight: 900, color: sylTone.fg }}>
                              {isPrimary ? 'ˈPRIMARY STRESS' : `SYLLABLE ${sIdx + 1}`}
                            </span>
                            <span
                              style={{
                                fontSize: isPrimary ? 18 : 15,
                                fontWeight: 900,
                                color: sylTone.fg,
                                fontFamily: "'Lucida Sans Unicode', 'DejaVu Sans', monospace",
                              }}
                            >
                              /{syl.syllable_ipa}/
                            </span>
                            <span style={{ fontSize: 10, fontWeight: 800, color: sylTone.fg, opacity: 0.85 }}>
                              {syl.duration_ms ?? 80}ms · {syl.score ?? 90}%
                            </span>
                          </Raised>
                        );
                      })}
                    </div>

                    {stressInfo.coaching_tip && (
                      <div style={{ fontSize: 12, fontWeight: 700, color: card.text }}>
                        🎯 <strong>Word Stress Tip:</strong> {stressInfo.coaching_tip}
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* PHONEME-BY-PHONEME ARTICULATORY DIFF CARDS */}
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(215px, 1fr))',
                  gap: 12,
                }}
              >
                {selectedWord.phonemes.map((p, idx) => {
                  const item: PhonemeItem =
                    typeof p === 'string'
                      ? { expected: p, heard: p, gop_score: 100, error: null }
                      : p;
                  const isOk = !item.error;
                  const diffText =
                    item.articulatory_diff ||
                    (!isOk ? computeClientArticulatoryDiff(item.expected, item.heard) : null);
                  const tip = item.expected ? PHONEME_TIPS[item.expected] : undefined;

                  return (
                    <GameCard
                      key={idx}
                      borderColor={isOk ? '#16A34A' : '#DC2626'}
                      depth={3}
                      innerStyle={{ padding: 12 }}
                      onClick={() => {
                        const v = ipaToViseme(item.expected);
                        avatarRuntimeRef.current?.setMouth(v);
                        setCurrentViseme(`${v} (/${item.expected || ''}/)`);
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span
                          style={{
                            fontSize: 18,
                            fontWeight: 900,
                            fontFamily: "'Lucida Sans Unicode', 'DejaVu Sans', monospace",
                          }}
                        >
                          /{item.expected || '∅'}/ {!isOk ? `→ /${item.heard || '∅'}/` : ''}
                        </span>
                        <Pill
                          label={`${item.gop_score}%`}
                          tone={isOk ? 'green' : 'red'}
                        />
                      </div>
                      <div style={{ fontSize: 11, fontWeight: 800, color: card.muted, marginTop: 4 }}>
                        {isOk ? 'ACCURATE SOUND' : `ERROR: ${item.error?.toUpperCase()}`}
                      </div>
                      {diffText && !isOk && (
                        <div
                          style={{
                            fontSize: 11,
                            fontWeight: 700,
                            color: card.text,
                            backgroundColor: card.surface,
                            borderLeft: '3px solid #F89A1C',
                            padding: '6px 8px',
                            borderRadius: 6,
                            marginTop: 8,
                            lineHeight: 1.35,
                          }}
                        >
                          🔬 <strong>Articulatory Diff:</strong> {diffText}
                        </div>
                      )}
                      {tip && (
                        <div style={{ fontSize: 11, color: card.muted, marginTop: 6, lineHeight: 1.35 }}>
                          💡 <strong>Mouth Placement:</strong> {tip}
                        </div>
                      )}
                    </GameCard>
                  );
                })}
              </div>
            </GameCard>
          )}

          {/* CONNECTED SPEECH, WORD LINKING (‿) & RHYTHM CARD */}
          {(() => {
            const conn =
              assessment.connected_speech || computeClientConnectedSpeech(assessment.words);
            if (!conn || conn.links.length === 0) return null;
            return (
              <GameCard>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    gap: 10,
                    marginBottom: 12,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span style={{ fontSize: 17, fontWeight: 900 }}>
                      CONNECTED SPEECH & WORD LINKING (‿)
                    </span>
                    <Pill label={`LINKING SCORE: ${conn.linking_score}%`} tone="green" />
                    <Pill
                      label={`INTONATION: ${conn.intonation_type.replace('_', ' ').toUpperCase()}`}
                      tone="blue"
                    />
                  </div>
                </div>

                <div
                  style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: card.text,
                    backgroundColor: card.surface,
                    border: `2px solid ${card.border}`,
                    borderRadius: 12,
                    padding: '10px 14px',
                    marginBottom: 12,
                  }}
                >
                  🔗 {conn.summary_tip}
                </div>

                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(225px, 1fr))',
                    gap: 10,
                  }}
                >
                  {conn.links.map((lk, lIdx) => {
                    const isSmooth = lk.status === 'smooth_link';
                    const isChoppy = lk.status === 'choppy_pause';
                    const pillTone = isSmooth ? 'green' : isChoppy ? 'red' : 'amber';
                    const typeLabel =
                      lk.link_type === 'consonant_to_vowel'
                        ? 'C→V LIAISON'
                        : lk.link_type === 'vowel_to_vowel'
                        ? 'V→V GLIDE'
                        : lk.link_type === 'consonant_cluster'
                        ? 'GEMINATE LINK'
                        : 'WORD TRANSITION';
                    return (
                      <div
                        key={lIdx}
                        style={{
                          backgroundColor: card.surface,
                          border: `2px solid ${isChoppy ? '#DC2626' : card.border}`,
                          borderRadius: 14,
                          padding: 12,
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: 15, fontWeight: 900 }}>
                            {lk.from_word} <span style={{ color: '#F89A1C' }}>‿</span> {lk.to_word}
                          </span>
                          <Pill label={`${lk.gap_ms}ms`} tone={pillTone} />
                        </div>
                        <div style={{ fontSize: 10, fontWeight: 800, color: card.muted, marginTop: 4 }}>
                          {typeLabel} · {lk.status.replace('_', ' ').toUpperCase()}
                        </div>
                        {lk.coaching_tip && (
                          <div style={{ fontSize: 11, color: card.text, marginTop: 6, lineHeight: 1.35 }}>
                            {lk.coaching_tip}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </GameCard>
            );
          })()}
        </div>
      )}

      {/* 6. GTS ACADEMY ASSESSMENT RESULT POPUP MODAL */}
      {showResultPopup && assessment && (
        <div
          onClick={() => setShowResultPopup(false)}
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(15, 23, 42, 0.72)',
            backdropFilter: 'blur(8px)',
            zIndex: 200,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 16,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{ width: '100%', maxWidth: 560 }}
          >
            <GameCard
              borderColor={
                assessment.overall_score >= 85
                  ? '#16A34A'
                  : assessment.overall_score >= 65
                  ? '#F89A1C'
                  : '#DC2626'
              }
              depth={7}
              innerStyle={{
                padding: '24px 22px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 18,
              }}
            >
              {/* Top Badges & Stars */}
              <div
                style={{
                  width: '100%',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: 8,
                }}
              >
                <Pill
                  label={
                    assessment.overall_score >= 92
                      ? 'RANK S · MASTER'
                      : assessment.overall_score >= 82
                      ? 'RANK A · GREAT'
                      : assessment.overall_score >= 68
                      ? 'RANK B · GOOD'
                      : 'RANK C · PRACTICE'
                  }
                  tone={assessment.overall_score >= 82 ? 'green' : 'orange'}
                />
                <StarRow
                  earned={
                    assessment.overall_score >= 90 ? 3 : assessment.overall_score >= 75 ? 2 : 1
                  }
                  total={3}
                  size={26}
                />
                <Pill
                  label={`+${
                    (assessment.overall_score >= 90 ? 3 : assessment.overall_score >= 75 ? 2 : 1) *
                    20
                  } XP`}
                  tone="navy"
                  icon={<Zap size={12} />}
                />
              </div>

              {/* Headline */}
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 22, fontWeight: 900, color: card.text }}>
                  {assessment.overall_score >= 90
                    ? '🌟 Outstanding Pronunciation!'
                    : assessment.overall_score >= 75
                    ? '👏 Great Job — Almost Native!'
                    : '💪 Good Try — Let’s Polish the Red Words!'}
                </div>
                <div style={{ fontSize: 13, fontWeight: 700, color: card.muted, marginTop: 2 }}>
                  Scored in {latencyMs ?? 0}ms · Single-Pass Wav2Vec2 ONNX
                </div>
              </div>

              {/* Centerpiece Score Ring & Sub-Metrics */}
              <div
                className="gts-modal-score-grid"
                style={{
                  backgroundColor: card.surface,
                  border: `2px solid ${card.border}`,
                }}
              >
                <ProgressRing
                  size={104}
                  stroke={10}
                  percent={assessment.overall_score}
                  color={
                    assessment.overall_score >= 85
                      ? '#16A34A'
                      : assessment.overall_score >= 65
                      ? '#F89A1C'
                      : '#DC2626'
                  }
                  trackColor={card.border}
                >
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 24, fontWeight: 900, lineHeight: 1 }}>
                      {assessment.overall_score}%
                    </div>
                    <div style={{ fontSize: 10, fontWeight: 800, color: card.muted, marginTop: 2 }}>
                      OVERALL
                    </div>
                  </div>
                </ProgressRing>

                <div className="gts-modal-bars" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {[
                    {
                      label: 'Phoneme Accuracy',
                      val: assessment.metrics.accuracy,
                      color: '#0D3B84',
                    },
                    {
                      label: 'Fluency & Rhythm',
                      val: assessment.metrics.fluency,
                      color: '#F89A1C',
                    },
                    {
                      label: 'Prosody Match',
                      val: assessment.metrics.prosody_similarity,
                      color: '#16A34A',
                    },
                  ].map((m) => (
                    <div key={m.label}>
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          fontSize: 12,
                          fontWeight: 800,
                          marginBottom: 4,
                        }}
                      >
                        <span>{m.label}</span>
                        <span>{m.val}%</span>
                      </div>
                      <div
                        style={{
                          height: 8,
                          backgroundColor: card.border,
                          borderRadius: 999,
                          overflow: 'hidden',
                        }}
                      >
                        <div
                          style={{
                            width: `${m.val}%`,
                            height: '100%',
                            backgroundColor: m.color,
                            borderRadius: 999,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Word-by-Word Summary Strip inside Popup */}
              <div style={{ width: '100%' }}>
                <div
                  style={{
                    fontSize: 11,
                    fontWeight: 900,
                    color: card.muted,
                    marginBottom: 8,
                    textTransform: 'uppercase',
                  }}
                >
                  WORD BREAKDOWN (TAP A WORD TO INSPECT)
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {assessment.words.map((w, idx) => {
                    const tone =
                      w.status === 'correct'
                        ? getGameTone('green', isDark)
                        : w.status === 'warning'
                        ? getGameTone('orange', isDark)
                        : getGameTone('rose', isDark);
                    return (
                      <Raised
                        key={idx}
                        face={tone.face}
                        shelf={tone.shelf}
                        radius={12}
                        depth={3}
                        onClick={() => {
                          setSelectedWordIdx(idx);
                          setShowResultPopup(false);
                        }}
                        innerStyle={{
                          padding: '6px 12px',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 6,
                        }}
                      >
                        <span style={{ fontWeight: 900, fontSize: 14, color: tone.fg }}>
                          {w.word}
                        </span>
                        <span style={{ fontWeight: 800, fontSize: 11, color: tone.fg, opacity: 0.9 }}>
                          {w.score}%
                        </span>
                      </Raised>
                    );
                  })}
                </div>
              </div>

              {/* Popup Footer Action Buttons */}
              <div className="gts-modal-footer">
                {audioBlobUrl && (
                  <ChunkyButton
                    tone="secondary"
                    size="sm"
                    label={isPlayingUserVoice ? 'Playing...' : 'Play My Voice'}
                    icon={<Play size={15} />}
                    onClick={playMyVoiceSynced}
                  />
                )}
                <div className="gts-modal-footer-right" style={{ display: 'flex', gap: 10, marginLeft: 'auto' }}>
                  <ChunkyButton
                    tone="accent"
                    size="sm"
                    label="Try Again"
                    icon={<Mic size={15} />}
                    onClick={() => {
                      setShowResultPopup(false);
                      void toggleRecording(false);
                    }}
                  />
                  <ChunkyButton
                    tone="primary"
                    size="sm"
                    label="Inspect Details"
                    onClick={() => setShowResultPopup(false)}
                  />
                </div>
              </div>
            </GameCard>
          </div>
        </div>
      )}
    </div>
  );
}

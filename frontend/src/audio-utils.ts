/**
 * Client-Side Audio Standards (Section 4 of Master Engineering Handoff):
 * - Downsamples microphone float32 audio to 16kHz mono
 * - Trims leading/trailing silence on the client before upload
 * - Encodes to 16kHz, 16-bit, mono PCM WAV
 */

export function mergeBuffers(chunks: Float32Array[]): Float32Array {
  const total = chunks.reduce((acc, c) => acc + c.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

export function downsampleBuffer(
  buffer: Float32Array,
  inRate: number,
  outRate: number = 16000
): Float32Array {
  if (inRate === outRate) return buffer;
  const ratio = inRate / outRate;
  const newLen = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(buffer.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    let count = 0;
    for (let j = start; j < end; j++) {
      sum += buffer[j];
      count++;
    }
    result[i] = count > 0 ? sum / count : buffer[start];
  }
  return result;
}

export function trimSilenceClientSide(
  samples: Float32Array,
  sampleRate: number = 16000
): Float32Array {
  const frameLen = Math.floor(sampleRate * 0.02);
  const numFrames = Math.floor(samples.length / frameLen);
  if (numFrames <= 2) return samples;

  let peakRms = 0;
  const rmsArr = new Float32Array(numFrames);
  for (let f = 0; f < numFrames; f++) {
    let sumSq = 0;
    const base = f * frameLen;
    for (let i = 0; i < frameLen; i++) {
      const v = samples[base + i];
      sumSq += v * v;
    }
    const rms = Math.sqrt(sumSq / frameLen);
    rmsArr[f] = rms;
    if (rms > peakRms) peakRms = rms;
  }
  if (peakRms < 1e-4) return samples;

  const thresh = peakRms * 0.025;
  let firstActive = 0;
  let lastActive = numFrames - 1;
  while (firstActive < numFrames && rmsArr[firstActive] < thresh) firstActive++;
  while (lastActive > firstActive && rmsArr[lastActive] < thresh) lastActive--;

  const pad = Math.floor(sampleRate * 0.06);
  const start = Math.max(0, firstActive * frameLen - pad);
  const end = Math.min(samples.length, (lastActive + 1) * frameLen + pad);
  return samples.slice(start, end);
}

export function encodeMono16kWav(samples: Float32Array, sampleRate: number = 16000): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

/** Synthesize a clean Web Audio sine chime for UI & recording cues. */
export function playTone(freq: number, durationSec: number = 0.08): void {
  try {
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.14, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + durationSec);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + durationSec);
    window.setTimeout(() => {
      void ctx.close().catch(() => {});
    }, durationSec * 1000 + 120);
  } catch {
    // Ignore audio context blocks
  }
}

/** Play 3-note or 4-note ascending victory fanfare on 2-star / 3-star clears. */
export function playVictoryChime(isThreeStar: boolean): void {
  const notes = isThreeStar ? [523.25, 659.25, 783.99, 1046.5] : [523.25, 659.25, 783.99];
  notes.forEach((n, idx) => {
    window.setTimeout(() => playTone(n, 0.16), idx * 95);
  });
}

/** Trigger a celebration confetti burst on 3-star clears. */
export function launchConfetti(canvas: HTMLCanvasElement | null): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  const colors = ['#0D3B84', '#F89A1C', '#16A34A', '#38BDF8', '#A855F7'];
  const particles = Array.from({ length: 75 }, () => ({
    x: canvas.width / 2 + (Math.random() - 0.5) * 260,
    y: canvas.height * 0.35,
    vx: (Math.random() - 0.5) * 14,
    vy: -Math.random() * 12 - 4,
    size: Math.random() * 8 + 5,
    color: colors[Math.floor(Math.random() * colors.length)],
    rot: Math.random() * 6.28,
    vrot: (Math.random() - 0.5) * 0.25,
  }));

  let frame = 0;
  function step() {
    frame++;
    ctx!.clearRect(0, 0, canvas!.width, canvas!.height);
    particles.forEach((p) => {
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.35;
      p.rot += p.vrot;
      ctx!.save();
      ctx!.translate(p.x, p.y);
      ctx!.rotate(p.rot);
      ctx!.fillStyle = p.color;
      ctx!.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
      ctx!.restore();
    });
    if (frame < 95) requestAnimationFrame(step);
    else ctx!.clearRect(0, 0, canvas!.width, canvas!.height);
  }
  step();
}

const VOWEL_SET = new Set('aeiouɑɒæɛɪɔʊʌəɐɜɝɚɨʉɵ'.split(''));
const VOICED_SET = new Set('bdɡgvðzʒdʒmnŋlɹrwj'.split(''));
const STOP_SET = new Set('ptkbdɡgʔ'.split(''));
const FRICATIVE_SET = new Set('fvθðszʃʒh'.split(''));

function isVowelIpa(sym: string): boolean {
  return Array.from(sym).some((c) => VOWEL_SET.has(c));
}

/**
 * Compute learner-friendly articulatory feature diff between expected and heard IPA
 * (used when the backend response comes from an older container without articulatory_diff).
 */
export function computeClientArticulatoryDiff(expected?: string | null, heard?: string | null): string {
  if (!expected && heard) {
    return `Extra /${heard}/ sound inserted — transition smoothly without adding /${heard}/.`;
  }
  if (expected && !heard) {
    return `Skipped /${expected}/ sound — articulate /${expected}/ clearly before moving to the next sound.`;
  }
  if (!expected || !heard || expected === heard) {
    return 'Mouth & tongue placement matched naturally.';
  }

  const cues: string[] = [];
  if (FRICATIVE_SET.has(expected) && STOP_SET.has(heard)) {
    cues.push(
      `Airflow was blocked completely (/${heard}/ stop); keep a narrow gap so air flows continuously for /${expected}/`
    );
  } else if (STOP_SET.has(expected) && FRICATIVE_SET.has(heard)) {
    cues.push(
      `Air leaked continuously (/${heard}/ fricative); make a crisp complete seal then release for /${expected}/`
    );
  }

  if ((expected === 'θ' || expected === 'ð') && ['t', 'd', 's', 'z'].includes(heard)) {
    cues.push(
      'Move your tongue tip slightly forward so it rests lightly between your upper and lower front teeth'
    );
  } else if ((expected === 'ʃ' || expected === 'ʒ') && (heard === 's' || heard === 'z')) {
    cues.push(
      "Pull your tongue tip slightly back toward the roof of your mouth and round your lips ('shh')"
    );
  } else if ((expected === 's' || expected === 'z') && ['ʃ', 'θ', 't'].includes(heard)) {
    cues.push(
      "Keep your tongue tip just behind your upper front teeth (without touching between teeth) for a crisp 'sss'"
    );
  }

  const expVoiced = VOICED_SET.has(expected) || isVowelIpa(expected);
  const hrdVoiced = VOICED_SET.has(heard) || isVowelIpa(heard);
  if (expVoiced && !hrdVoiced) {
    cues.push(
      `Turn on your vocal cords (feel your throat buzz) to voice /${expected}/ instead of voiceless /${heard}/`
    );
  } else if (!expVoiced && hrdVoiced) {
    cues.push(`Use pure breath without throat vibration for unvoiced /${expected}/`);
  }

  if (isVowelIpa(expected) && isVowelIpa(heard)) {
    if (expected.includes('ː') && !heard.includes('ː')) {
      cues.push(`Hold vowel /${expected}/ slightly longer with firmer tongue tension`);
    } else if (expected === 'æ' && heard !== 'æ') {
      cues.push('Drop your jaw wider and keep your tongue low in the front of your mouth for /æ/');
    } else {
      cues.push(`Adjust vowel tongue height and lip rounding from /${heard}/ toward /${expected}/`);
    }
  }

  return cues.length > 0
    ? cues.join(' · ') + '.'
    : `Produced /${heard}/ instead of /${expected}/ — isolate /${expected}/ slowly and match the Coach mouth shape.`;
}

/**
 * Fallback syllable & lexical stress analyzer when talking to an older backend container
 * that hasn't been restarted with `word_stress` yet.
 */
export function computeClientWordStress(
  expectedIpa: string,
  phonemes: Array<{ expected: string | null; gop_score?: number } | string>,
  startMs = 0,
  endMs = 240
) {
  const syms = phonemes.map((p) => (typeof p === 'string' ? p : p.expected || '')).filter(Boolean);
  const scores = phonemes.map((p) => (typeof p === 'string' ? 100 : p.gop_score ?? 100));
  const vowelIdx: number[] = [];
  syms.forEach((s, i) => {
    if (isVowelIpa(s)) vowelIdx.push(i);
  });

  const totalDur = Math.max(80, endMs - startMs);
  if (vowelIdx.length <= 1) {
    return {
      syllable_count: 1,
      expected_primary_index: 0,
      detected_peak_index: 0,
      stress_correct: true,
      stress_pattern_display: `ˈ${expectedIpa}`,
      coaching_tip: null,
      syllables: [
        {
          syllable_ipa: expectedIpa,
          stress: 'primary' as const,
          duration_ms: totalDur,
          score: scores.length ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length) : 100,
        },
      ],
    };
  }

  // Group phonemes around vowel nuclei
  const syllables: Array<{
    syllable_ipa: string;
    stress: 'primary' | 'secondary' | 'unstressed';
    duration_ms: number;
    score: number;
  }> = [];
  let startP = 0;
  // Heuristic for English multi-syllable stress when raw stress mark isn't present:
  // Words ending in -tion / -sion stress the penultimate syllable; otherwise stress non-schwa vowel
  let primaryIdx = 0;
  for (let s = 0; s < vowelIdx.length; s++) {
    const vSym = syms[vowelIdx[s]];
    if (vSym !== 'ə' && vSym !== 'ɐ' && vSym !== 'ɪ') {
      primaryIdx = s;
      break;
    }
  }
  if (expectedIpa.endsWith('ʃən') && vowelIdx.length >= 2) {
    primaryIdx = vowelIdx.length - 2;
  }

  for (let s = 0; s < vowelIdx.length; s++) {
    const v = vowelIdx[s];
    const endP =
      s === vowelIdx.length - 1
        ? syms.length
        : Math.max(v + 1, vowelIdx[s + 1] - 1);
    const sliceSyms = syms.slice(startP, endP);
    const sliceScores = scores.slice(startP, endP);
    const avgScore = sliceScores.length
      ? Math.round(sliceScores.reduce((a, b) => a + b, 0) / sliceScores.length)
      : 85;
    const durMs = Math.round((totalDur * Math.max(1, sliceSyms.length)) / Math.max(1, syms.length));
    syllables.push({
      syllable_ipa: sliceSyms.join(''),
      stress: s === primaryIdx ? 'primary' : 'unstressed',
      duration_ms: durMs,
      score: avgScore,
    });
    startP = endP;
  }

  let peakIdx = 0;
  syllables.forEach((syl, idx) => {
    if (syl.score * syl.duration_ms > syllables[peakIdx].score * syllables[peakIdx].duration_ms) {
      peakIdx = idx;
    }
  });
  const stressOk = peakIdx === primaryIdx || syllables[primaryIdx].score >= 80;
  const patternStr = syllables
    .map((s, idx) => (idx === primaryIdx ? `ˈ${s.syllable_ipa.toUpperCase()}` : s.syllable_ipa))
    .join(' · ');

  return {
    syllable_count: syllables.length,
    expected_primary_index: primaryIdx,
    detected_peak_index: peakIdx,
    stress_correct: stressOk,
    stress_pattern_display: patternStr,
    coaching_tip: stressOk
      ? `Good primary stress emphasis on syllable #${primaryIdx + 1} (/${syllables[primaryIdx].syllable_ipa}/ in ${patternStr}).`
      : `Put stronger emphasis and longer vowel length on syllable #${primaryIdx + 1} (/${syllables[primaryIdx].syllable_ipa}/ in ${patternStr}).`,
    syllables,
  };
}

/**
 * Fallback connected-speech linking analyzer between adjacent words.
 */
export function computeClientConnectedSpeech(
  words: Array<{
    word: string;
    expected_ipa: string;
    start_ms?: number;
    end_ms?: number;
  }>
) {
  const links: Array<{
    from_word: string;
    to_word: string;
    link_type: 'consonant_to_vowel' | 'consonant_cluster' | 'vowel_to_vowel' | 'normal';
    gap_ms: number;
    status: 'smooth_link' | 'natural_pause' | 'choppy_pause';
    coaching_tip?: string | null;
  }> = [];

  let smoothCount = 0;
  let choppyCount = 0;

  for (let i = 0; i < words.length - 1; i++) {
    const w1 = words[i];
    const w2 = words[i + 1];
    const gapMs = Math.max(0, (w2.start_ms ?? 0) - (w1.end_ms ?? 0));
    const endCh = (w1.expected_ipa || '').slice(-1);
    const startCh = (w2.expected_ipa || '').slice(0, 1);
    const endVowel = isVowelIpa(endCh);
    const startVowel = isVowelIpa(startCh);

    const linkType =
      !endVowel && startVowel
        ? 'consonant_to_vowel'
        : endVowel && startVowel
        ? 'vowel_to_vowel'
        : !endVowel && !startVowel && endCh === startCh
        ? 'consonant_cluster'
        : 'normal';

    let status: 'smooth_link' | 'natural_pause' | 'choppy_pause' = 'smooth_link';
    let tip: string | null = null;
    if (gapMs <= 90) {
      status = 'smooth_link';
      smoothCount++;
      if (linkType === 'consonant_to_vowel') {
        tip = `Smooth C→V link: "${w1.word}‿${w2.word}" connected seamlessly (${gapMs}ms).`;
      }
    } else if (gapMs <= 190) {
      status = 'natural_pause';
      if (linkType === 'consonant_to_vowel') {
        tip = `Link the final /${endCh}/ of "${w1.word}" directly into the vowel of "${w2.word}" ("${w1.word}‿${w2.word}").`;
      }
    } else {
      status = 'choppy_pause';
      choppyCount++;
      tip = `Choppy break (${gapMs}ms) between "${w1.word}" and "${w2.word}" — blend "${w1.word}‿${w2.word}" in one continuous breath.`;
    }

    links.push({
      from_word: w1.word,
      to_word: w2.word,
      link_type: linkType,
      gap_ms: gapMs,
      status,
      coaching_tip: tip,
    });
  }

  const total = Math.max(1, links.length);
  const linkingScore = links.length
    ? Math.round((100 * (smoothCount + 0.65 * (links.length - smoothCount - choppyCount))) / total)
    : 100;

  return {
    linking_score: linkingScore,
    smooth_links_count: smoothCount,
    choppy_Pause_count: choppyCount,
    intonation_contour_hz: [],
    intonation_type: 'expressive' as const,
    summary_tip:
      choppyCount > 0
        ? `Detected ${choppyCount} choppy break(s) between words. Practice linking consonant-to-vowel boundaries ("‿") smoothly.`
        : 'Natural connected speech flow and smooth word-to-word transitions!',
    links,
  };
}

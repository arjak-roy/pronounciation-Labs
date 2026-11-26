import React, { createContext, useContext, useState, type CSSProperties, type ReactNode } from 'react';

/*
 * Exact web implementation of gts-candidate-app's Academy Game UI Kit
 * (`src/academy/features/programs/components/program-detail/game-ui.tsx`)
 * Tokens:
 *   gts-primary: #0D3B84 (shelf #082A60)
 *   gts-orange:  #F89A1C (shelf #B86E0B)
 *   dm-canvas:   #0F172A, dm-surface: #1E293B, dm-card: #253348, dm-line: #2E3F5C, dm-primary: #7AA2FF
 */

export type ThemeMode = 'light' | 'dark';

interface ThemeContextValue {
  isDark: boolean;
  mode: ThemeMode;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  isDark: false,
  mode: 'light',
  toggleTheme: () => {},
});

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<ThemeMode>('light');
  const isDark = mode === 'dark';
  const toggleTheme = () => setMode((m) => (m === 'light' ? 'dark' : 'light'));
  return (
    <ThemeContext.Provider value={{ isDark, mode, toggleTheme }}>
      <div
        style={{
          minHeight: '100vh',
          backgroundColor: isDark ? '#0F172A' : '#F8FAFC',
          color: isDark ? '#F1F5F9' : '#0F172A',
          fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
          transition: 'background-color 0.2s ease, color 0.2s ease',
        }}
      >
        {children}
      </div>
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}

export type GameTone =
  | 'navy'
  | 'orange'
  | 'green'
  | 'blue'
  | 'gray'
  | 'purple'
  | 'red'
  | 'white'
  | 'soft'
  | 'rose'
  | 'teal';

export interface ToneColors {
  face: string;
  shelf: string;
  fg: string;
}

export const TONES: Record<GameTone, { light: ToneColors; dark: ToneColors }> = {
  navy: {
    light: { face: '#0D3B84', shelf: '#082A60', fg: '#FFFFFF' },
    dark: { face: '#2F5FC4', shelf: '#1E3F87', fg: '#FFFFFF' },
  },
  orange: {
    light: { face: '#F89A1C', shelf: '#B86E0B', fg: '#1F1300' },
    dark: { face: '#F89A1C', shelf: '#9A5A06', fg: '#1F1300' },
  },
  green: {
    light: { face: '#16A34A', shelf: '#15803D', fg: '#FFFFFF' },
    dark: { face: '#15803D', shelf: '#0F5E2C', fg: '#FFFFFF' },
  },
  blue: {
    light: { face: '#DBEAFE', shelf: '#BFD3F5', fg: '#1D4ED8' },
    dark: { face: '#172554', shelf: '#0F1B40', fg: '#93C5FD' },
  },
  gray: {
    light: { face: '#E2E8F0', shelf: '#CBD5E1', fg: '#64748B' },
    dark: { face: '#334155', shelf: '#1E293B', fg: '#94A3B8' },
  },
  purple: {
    light: { face: '#EDE9FE', shelf: '#D8CFFB', fg: '#6D28D9' },
    dark: { face: '#2E1065', shelf: '#1E0A45', fg: '#C4B5FD' },
  },
  red: {
    light: { face: '#B91C1C', shelf: '#7F1414', fg: '#FFFFFF' },
    dark: { face: '#B91C1C', shelf: '#6B1010', fg: '#FFFFFF' },
  },
  white: {
    light: { face: '#FFFFFF', shelf: '#E2E8F0', fg: '#0D3B84' },
    dark: { face: '#253348', shelf: '#1B2638', fg: '#7AA2FF' },
  },
  soft: {
    light: { face: '#F1F5F9', shelf: '#DCE3EC', fg: '#334155' },
    dark: { face: '#1E293B', shelf: '#141D2E', fg: '#CBD5E1' },
  },
  rose: {
    light: { face: '#FEE2E2', shelf: '#FBC9C9', fg: '#B91C1C' },
    dark: { face: '#3F1518', shelf: '#2A0E10', fg: '#FCA5A5' },
  },
  teal: {
    light: { face: '#CCFBF1', shelf: '#A7EDE0', fg: '#0F766E' },
    dark: { face: '#083D38', shelf: '#052925', fg: '#5EEAD4' },
  },
};

export function useGameTone(tone: GameTone): ToneColors {
  const { isDark } = useTheme();
  return isDark ? TONES[tone].dark : TONES[tone].light;
}

export function getGameTone(tone: GameTone, isDark: boolean): ToneColors {
  return isDark ? TONES[tone].dark : TONES[tone].light;
}

export function useCardColors() {
  const { isDark } = useTheme();
  return isDark
    ? { face: '#253348', card: '#253348', border: '#2E3F5C', shelf: '#1B2638', text: '#F1F5F9', muted: '#94A3B8', surface: '#1E293B' }
    : { face: '#FFFFFF', card: '#FFFFFF', border: '#E2E8F0', shelf: '#E2E8F0', text: '#0F172A', muted: '#64748B', surface: '#F8FAFC' };
}

export interface RaisedProps {
  face: string;
  shelf: string;
  radius: number;
  depth?: number;
  border?: string;
  dashed?: boolean;
  style?: CSSProperties;
  innerStyle?: CSSProperties;
  children?: ReactNode;
  onClick?: () => void;
}

/** Signature GTS Academy two-layer chunky surface sitting on a darker shelf. */
export function Raised({
  face,
  shelf,
  radius,
  depth = 5,
  border,
  dashed,
  style,
  innerStyle,
  children,
  onClick,
}: RaisedProps) {
  return (
    <div
      onClick={onClick}
      style={{
        backgroundColor: shelf,
        borderRadius: radius,
        paddingBottom: depth,
        cursor: onClick ? 'pointer' : undefined,
        userSelect: onClick ? 'none' : undefined,
        ...style,
      }}
    >
      <div
        style={{
          backgroundColor: face,
          borderRadius: radius,
          border: border ? `2px ${dashed ? 'dashed' : 'solid'} ${border}` : undefined,
          ...innerStyle,
        }}
      >
        {children}
      </div>
    </div>
  );
}

/** GTS Academy GameCard: rounded-3xl (22px) card sitting on a 5px shelf. */
export function GameCard({
  children,
  style,
  innerStyle,
  borderColor,
  dashed,
  depth = 5,
  onClick,
}: {
  children: ReactNode;
  style?: CSSProperties;
  innerStyle?: CSSProperties;
  borderColor?: string;
  dashed?: boolean;
  depth?: number;
  onClick?: () => void;
}) {
  const card = useCardColors();
  return (
    <Raised
      face={card.face}
      shelf={borderColor && !dashed ? borderColor : card.shelf}
      border={borderColor ?? card.border}
      dashed={dashed}
      radius={22}
      depth={dashed ? 0 : depth}
      style={style}
      innerStyle={{ padding: 18, ...innerStyle }}
      onClick={onClick}
    >
      {children}
    </Raised>
  );
}

/** GTS Academy ChunkyButton: raised button that physically sinks onto its shelf when pressed. */
export function ChunkyButton({
  label,
  children,
  onClick,
  tone = 'primary',
  icon,
  disabled,
  loading,
  size = 'md',
  fullWidth,
  type = 'button',
  title,
}: {
  label?: string;
  children?: ReactNode;
  onClick?: () => void;
  tone?: 'primary' | 'accent' | 'secondary' | 'danger' | 'green' | 'navy' | 'orange' | 'soft' | 'red';
  icon?: ReactNode;
  disabled?: boolean;
  loading?: boolean;
  size?: 'sm' | 'md' | 'lg';
  fullWidth?: boolean;
  type?: 'button' | 'submit';
  title?: string;
}) {
  const { isDark } = useTheme();
  const card = useCardColors();
  const [pressed, setPressed] = useState(false);

  const colors =
    tone === 'accent' || tone === 'orange'
      ? getGameTone('orange', isDark)
      : tone === 'secondary' || tone === 'soft'
      ? { face: card.face, shelf: card.shelf, fg: isDark ? '#F1F5F9' : '#0F172A' }
      : tone === 'danger' || tone === 'red'
      ? getGameTone('red', isDark)
      : tone === 'green'
      ? getGameTone('green', isDark)
      : getGameTone('navy', isDark);

  const muted = Boolean(disabled || loading);
  const height = size === 'lg' ? 52 : size === 'sm' ? 38 : 46;
  const depth = size === 'sm' ? 3 : 5;
  const sunk = pressed && !muted;
  const hasContent = Boolean(label || children);

  return (
    <button
      type={type}
      title={title}
      disabled={muted}
      onClick={onClick}
      onMouseDown={() => setPressed(true)}
      onMouseUp={() => setPressed(false)}
      onMouseLeave={() => setPressed(false)}
      onTouchStart={() => setPressed(true)}
      onTouchEnd={() => setPressed(false)}
      style={{
        appearance: 'none',
        background: 'transparent',
        border: 'none',
        padding: 0,
        paddingTop: sunk ? depth : 0,
        width: fullWidth ? '100%' : 'auto',
        opacity: disabled ? 0.55 : 1,
        cursor: muted ? 'not-allowed' : 'pointer',
        fontFamily: 'inherit',
      }}
    >
      <Raised
        face={colors.face}
        shelf={colors.shelf}
        border={tone === 'secondary' || tone === 'soft' ? card.border : undefined}
        radius={size === 'sm' ? 12 : 14}
        depth={sunk ? 0 : depth}
        innerStyle={{
          height,
          minWidth: height,
          paddingLeft: hasContent ? (size === 'sm' ? 14 : 20) : 0,
          paddingRight: hasContent ? (size === 'sm' ? 14 : 20) : 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 7,
          color: colors.fg,
          fontSize: size === 'lg' ? 15 : size === 'sm' ? 12 : 13,
          fontWeight: 900,
          letterSpacing: 0.3,
          whiteSpace: 'nowrap',
        }}
      >
        {icon}
        {label ? label.toUpperCase() : children}
      </Raised>
    </button>
  );
}

/** Square/rounded icon tile on its own shelf. */
export function IconTile({
  tone,
  size = 46,
  depth = 4,
  children,
}: {
  tone: GameTone;
  size?: number;
  depth?: number;
  children: ReactNode;
}) {
  const colors = useGameTone(tone);
  return (
    <Raised
      face={colors.face}
      shelf={colors.shelf}
      radius={Math.round(size / 3)}
      depth={depth}
      style={{ width: size, flexShrink: 0 }}
      innerStyle={{
        width: size,
        height: size,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: colors.fg,
      }}
    >
      {children}
    </Raised>
  );
}

/** Circular SVG Progress Ring matching gts-candidate-app. */
export function ProgressRing({
  size = 100,
  stroke = 10,
  percent,
  value,
  color,
  trackColor,
  tone,
  label,
  sublabel,
  children,
}: {
  size?: number;
  stroke?: number;
  percent?: number;
  value?: number;
  color?: string;
  trackColor?: string;
  tone?: GameTone;
  label?: string;
  sublabel?: string;
  children?: ReactNode;
}) {
  const { isDark } = useTheme();
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const rawPct = typeof percent === 'number' ? percent : typeof value === 'number' ? value : 0;
  const clamped = Math.max(0, Math.min(100, rawPct));
  const resolvedColor =
    color ||
    (tone === 'green'
      ? '#16A34A'
      : tone === 'orange'
      ? '#F89A1C'
      : tone === 'red'
      ? '#DC2626'
      : '#0D3B84');
  const resolvedTrack = trackColor || (isDark ? '#1E293B' : '#E2E8F0');
  return (
    <div
      style={{
        width: size,
        height: size,
        position: 'relative',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
      }}
    >
      <svg
        width={size}
        height={size}
        style={{ position: 'absolute', transform: 'rotate(-90deg)' }}
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={resolvedTrack}
          strokeWidth={stroke}
          fill="none"
        />
        {clamped > 0 && (
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={resolvedColor}
            strokeWidth={stroke}
            fill="none"
            strokeLinecap="round"
            strokeDasharray={`${(circumference * clamped) / 100} ${circumference}`}
            style={{ transition: 'stroke-dasharray 0.45s ease' }}
          />
        )}
      </svg>
      {children || (
        <div style={{ textAlign: 'center' }}>
          {label && <div style={{ fontSize: 22, fontWeight: 900 }}>{label}</div>}
          {sublabel && <div style={{ fontSize: 10, fontWeight: 800, opacity: 0.7 }}>{sublabel}</div>}
        </div>
      )}
    </div>
  );
}

const PILL_TONES = {
  navy: { light: ['#E8EEF8', '#0D3B84'], dark: ['#13233E', '#7AA2FF'] },
  green: { light: ['#DCFCE7', '#166534'], dark: ['#052E26', '#6EE7B7'] },
  orange: { light: ['#FFF1DE', '#8A4B00'], dark: ['#3A2416', '#FDBA74'] },
  blue: { light: ['#EFF6FF', '#1D4ED8'], dark: ['#172554', '#93C5FD'] },
  red: { light: ['#FEE2E2', '#991B1B'], dark: ['#3F1518', '#FCA5A5'] },
  purple: { light: ['#EDE9FE', '#5B21B6'], dark: ['#2E1065', '#C4B5FD'] },
  amber: { light: ['#FEF3C7', '#92400E'], dark: ['#3B2A0E', '#FCD34D'] },
  slate: { light: ['#F1F5F9', '#334155'], dark: ['#1E293B', '#CBD5E1'] },
  gray: { light: ['#F1F5F9', '#334155'], dark: ['#1E293B', '#CBD5E1'] },
} as const;

export type PillTone = keyof typeof PILL_TONES;

export function Pill({
  label,
  children,
  tone = 'slate',
  icon,
}: {
  label?: string;
  children?: ReactNode;
  tone?: PillTone;
  icon?: ReactNode;
}) {
  const { isDark } = useTheme();
  const [bg, fg] = isDark ? PILL_TONES[tone].dark : PILL_TONES[tone].light;
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
        backgroundColor: bg,
        color: fg,
        borderRadius: 999,
        padding: '4px 10px',
        fontSize: 11,
        fontWeight: 800,
        letterSpacing: 0.3,
        whiteSpace: 'nowrap',
      }}
    >
      {icon}
      {label ? label.toUpperCase() : children}
    </span>
  );
}

export function StatTile({
  tone,
  icon,
  value,
  label,
}: {
  tone: GameTone;
  icon?: ReactNode;
  value: string;
  label: string;
}) {
  const card = useCardColors();
  return (
    <GameCard innerStyle={{ padding: 14, display: 'flex', alignItems: 'center', gap: 12 }}>
      {icon && (
        <IconTile tone={tone} size={42} depth={3}>
          {icon}
        </IconTile>
      )}
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 20, fontWeight: 900, color: card.text, lineHeight: 1.15 }}>
          {value}
        </div>
        <div style={{ fontSize: 12, fontWeight: 700, color: card.muted }}>{label}</div>
      </div>
    </GameCard>
  );
}

export function StarRow({ earned, total = 3, size = 18 }: { earned: number; total?: number; size?: number }) {
  const { isDark } = useTheme();
  const emptyColor = isDark ? '#334155' : '#DCE3EC';
  return (
    <span style={{ display: 'inline-flex', gap: 2, alignItems: 'center' }}>
      {Array.from({ length: total }, (_, idx) => (
        <svg
          key={idx}
          width={size}
          height={size}
          viewBox="0 0 24 24"
          fill={idx < earned ? '#F5B301' : emptyColor}
        >
          <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
        </svg>
      ))}
    </span>
  );
}

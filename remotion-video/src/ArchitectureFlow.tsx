import { useCurrentFrame, useVideoConfig, interpolate, spring, AbsoluteFill, Sequence } from "remotion";

// Color palette
const COLORS = {
  bg: "#0f172a",
  card: "#1e293b",
  cardBorder: "#334155",
  visual: "#3b82f6",
  text: "#10b981",
  fusion: "#8b5cf6",
  attention: "#f59e0b",
  output: "#ef4444",
  white: "#f8fafc",
  gray: "#94a3b8",
  lightBlue: "#60a5fa",
  lightGreen: "#34d399",
  lightPurple: "#a78bfa",
  lightOrange: "#fbbf24",
  lightRed: "#f87171",
};

// Reusable animated box component
const Box = ({
  x,
  y,
  w,
  h,
  color,
  label,
  sublabel,
  opacity,
  scale,
  borderColor,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
  label: string;
  sublabel?: string;
  opacity: number;
  scale: number;
  borderColor?: string;
}) => (
  <div
    style={{
      position: "absolute",
      left: x,
      top: y,
      width: w,
      height: h,
      backgroundColor: color,
      borderRadius: 12,
      border: `2px solid ${borderColor || color}`,
      opacity,
      transform: `scale(${scale})`,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      fontSize: h < 50 ? 14 : 18,
      fontWeight: 600,
      color: COLORS.white,
      textAlign: "center",
      padding: "8px",
      boxShadow: `0 4px 20px ${color}40`,
      overflow: "hidden",
    }}
  >
    <span>{label}</span>
    {sublabel && (
      <span style={{ fontSize: 11, fontWeight: 400, color: COLORS.gray, marginTop: 4 }}>
        {sublabel}
      </span>
    )}
  </div>
);

// Animated arrow between two points
const Arrow = ({
  x1, y1, x2, y2, color, opacity, progress,
}: {
  x1: number; y1: number; x2: number; y2: number;
  color: string; opacity: number; progress: number;
}) => {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = Math.sqrt(dx * dx + dy * dy);
  const angle = Math.atan2(dy, dx) * (180 / Math.PI);
  const visibleLength = length * progress;

  return (
    <div
      style={{
        position: "absolute",
        left: x1,
        top: y1,
        width: visibleLength,
        height: 3,
        backgroundColor: color,
        opacity,
        transformOrigin: "0 0",
        transform: `rotate(${angle}deg)`,
        boxShadow: `0 0 8px ${color}`,
        borderRadius: 2,
      }}
    />
  );
};

const scaleIn = (fps: number, delay: number) =>
  spring({ fps, delay, config: { damping: 12 } });

const fadeIn = (frame: number, startFrame: number, duration: number = 20) =>
  interpolate(frame, [startFrame, startFrame + duration], [0, 1], { extrapolateRight: "clamp", extrapolateLeft: "clamp" });

export const ArchitectureFlow = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Timing constants (in frames)
  const TITLE_END = 90;
  const VISUAL_START = 90;
  const TEXT_START = 120;
  const FAE_START = 180;
  const ADAPTER_START = 210;
  const FUSION_START = 270;
  const OUTPUT_START = 360;
  const JEPA_START = 420;
  const LOSSES_START = 480;
  const RESULT_START = 540;

  // Layout constants
  const LEFT_X = 80;
  const RIGHT_X = 1200;
  const CENTER_X = 750;
  const TOP_Y = 140;

  // Title phase
  const titleOpacity = fadeIn(frame, 0, 30);
  const subtitleOpacity = fadeIn(frame, 40, 30);

  const isTitlePhase = frame < TITLE_END;

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.bg }}>
      {/* Background gradient */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "radial-gradient(ellipse at 50% 0%, #1e293b 0%, #0f172a 70%)",
          opacity: 0.6,
        }}
      />

      {/* ===== TITLE SLIDE ===== */}
      {isTitlePhase && (
        <>
          <div
            style={{
              position: "absolute",
              top: "40%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              textAlign: "center",
              opacity: titleOpacity,
            }}
          >
            <h1 style={{ fontSize: 56, fontWeight: 800, color: COLORS.white, marginBottom: 12, letterSpacing: -1 }}>
              VDT-TransZero
            </h1>
            <p style={{ fontSize: 24, color: COLORS.gray, fontWeight: 400 }}>
              Visual-Descriptive Text Transformer for
            </p>
            <p style={{ fontSize: 24, color: COLORS.gray, fontWeight: 400, marginTop: 4 }}>
              Generalized Zero-Shot Learning
            </p>
          </div>
          <div
            style={{
              position: "absolute",
              bottom: "12%",
              left: "50%",
              transform: "translateX(-50%)",
              opacity: subtitleOpacity,
            }}
          >
            <p style={{ fontSize: 18, color: COLORS.gray, fontWeight: 300 }}>
              CUB-200-2011 · CLIP ViT-L/14@336px · GPT-5.5 Prompts · H = 74.09
            </p>
          </div>
        </>
      )}

      {/* ===== MAIN ARCHITECTURE ===== */}
      {frame >= VISUAL_START && (
        <>
          {/* --- Visual Branch --- */}
          <Sequence from={VISUAL_START}>
            <Box
              x={LEFT_X}
              y={TOP_Y}
              w={220}
              h={80}
              color={COLORS.visual}
              label="🦜 CUB Bird Image"
              sublabel="336 × 336"
              opacity={fadeIn(frame, VISUAL_START)}
              scale={scaleIn(fps, VISUAL_START)}
            />
          </Sequence>

          <Sequence from={VISUAL_START + 30}>
            <Box
              x={LEFT_X}
              y={TOP_Y + 120}
              w={220}
              h={80}
              color={COLORS.visual}
              label="CLIP ViT-L/14"
              sublabel="Frozen Encoder · 576 patches"
              opacity={fadeIn(frame, VISUAL_START + 30)}
              scale={scaleIn(fps, VISUAL_START + 30)}
            />
          </Sequence>

          <Sequence from={VISUAL_START + 60}>
            <Box
              x={LEFT_X}
              y={TOP_Y + 240}
              w={220}
              h={80}
              color={COLORS.visual}
              label="Patch Features"
              sublabel="576 × 768-d"
              opacity={fadeIn(frame, VISUAL_START + 60)}
              scale={scaleIn(fps, VISUAL_START + 60)}
            />
          </Sequence>

          {/* Arrows: Visual */}
          <Arrow x1={LEFT_X + 110} y1={TOP_Y + 80} x2={LEFT_X + 110} y2={TOP_Y + 117} color={COLORS.lightBlue} opacity={fadeIn(frame, VISUAL_START + 30)} progress={1} />
          <Arrow x1={LEFT_X + 110} y1={TOP_Y + 200} x2={LEFT_X + 110} y2={TOP_Y + 237} color={COLORS.lightBlue} opacity={fadeIn(frame, VISUAL_START + 60)} progress={1} />

          {/* --- FAE Module --- */}
          <Sequence from={FAE_START}>
            <Box
              x={LEFT_X}
              y={TOP_Y + 360}
              w={220}
              h={90}
              color={COLORS.fusion}
              label="Feature Augmentation"
              sublabel="FAE + Geometry Encoding"
              borderColor={COLORS.lightPurple}
              opacity={fadeIn(frame, FAE_START)}
              scale={scaleIn(fps, FAE_START)}
            />
          </Sequence>

          <Arrow x1={LEFT_X + 110} y1={TOP_Y + 320} x2={LEFT_X + 110} y2={TOP_Y + 357} color={COLORS.lightPurple} opacity={fadeIn(frame, FAE_START)} progress={1} />

          {/* --- Text Branch --- */}
          <Sequence from={TEXT_START}>
            <Box
              x={RIGHT_X}
              y={TOP_Y}
              w={280}
              h={80}
              color={COLORS.text}
              label="GPT-5.5 Descriptions"
              sublabel="CLIP photo of {class}, ... ×7"
              opacity={fadeIn(frame, TEXT_START)}
              scale={scaleIn(fps, TEXT_START)}
            />
          </Sequence>

          <Sequence from={TEXT_START + 30}>
            <Box
              x={RIGHT_X}
              y={TOP_Y + 120}
              w={280}
              h={80}
              color={COLORS.text}
              label="CLIP Text Encoder"
              sublabel="Frozen · 768-d"
              opacity={fadeIn(frame, TEXT_START + 30)}
              scale={scaleIn(fps, TEXT_START + 30)}
            />
          </Sequence>

          <Sequence from={ADAPTER_START}>
            <Box
              x={RIGHT_X}
              y={TOP_Y + 240}
              w={280}
              h={80}
              color={COLORS.text}
              label="Text Adapter"
              sublabel="Learnable · 200 class prototypes"
              borderColor={COLORS.lightGreen}
              opacity={fadeIn(frame, ADAPTER_START)}
              scale={scaleIn(fps, ADAPTER_START)}
            />
          </Sequence>

          {/* Arrows: Text */}
          <Arrow x1={RIGHT_X + 140} y1={TOP_Y + 80} x2={RIGHT_X + 140} y2={TOP_Y + 117} color={COLORS.lightGreen} opacity={fadeIn(frame, TEXT_START + 30)} progress={1} />
          <Arrow x1={RIGHT_X + 140} y1={TOP_Y + 200} x2={RIGHT_X + 140} y2={TOP_Y + 237} color={COLORS.lightGreen} opacity={fadeIn(frame, ADAPTER_START)} progress={1} />

          {/* --- Cross-Modal Transformer --- */}
          <Sequence from={FUSION_START}>
            {/* S2V Path */}
            <div style={{
              position: "absolute",
              left: LEFT_X + 120, top: TOP_Y + 350,
              width: 260, height: 110,
              backgroundColor: COLORS.card,
              borderRadius: 12,
              border: "1px solid " + COLORS.cardBorder,
              opacity: fadeIn(frame, FUSION_START),
              transform: `scale(${scaleIn(fps, FUSION_START)})`,
              display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center",
              padding: 8,
            }}>
              <span style={{ fontSize: 15, fontWeight: 600, color: COLORS.attention }}>S → V (Visual Enhanced)</span>
              <span style={{ fontSize: 11, color: COLORS.gray, marginTop: 4 }}>
                Semantics query visual via Cross-Attention
              </span>
              <span style={{ fontSize: 10, color: COLORS.gray }}>
                Produces: score_s2v
              </span>
            </div>

            {/* V2S Path */}
            <div style={{
              position: "absolute",
              left: RIGHT_X - 80, top: TOP_Y + 350,
              width: 260, height: 110,
              backgroundColor: COLORS.card,
              borderRadius: 12,
              border: "1px solid " + COLORS.cardBorder,
              opacity: fadeIn(frame, FUSION_START),
              transform: `scale(${scaleIn(fps, FUSION_START)})`,
              display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center",
              padding: 8,
            }}>
              <span style={{ fontSize: 15, fontWeight: 600, color: COLORS.attention }}>V → S (Text Enhanced)</span>
              <span style={{ fontSize: 11, color: COLORS.gray, marginTop: 4 }}>
                Visual query semantics via Cross-Attention
              </span>
              <span style={{ fontSize: 10, color: COLORS.gray }}>
                Produces: score_v2s
              </span>
            </div>

            {/* Center: CrossModalTransformer */}
            <Box
              x={CENTER_X - 90}
              y={TOP_Y + 410}
              w={260}
              h={70}
              color={COLORS.attention}
              label="Bidirectional Cross-Modal"
              sublabel="Transformer (4 heads, 512-d)"
              borderColor={COLORS.lightOrange}
              opacity={fadeIn(frame, FUSION_START + 15)}
              scale={scaleIn(fps, FUSION_START + 15)}
            />

            {/* Arrows to center */}
            <Arrow x1={LEFT_X + 250} y1={TOP_Y + 410} x2={CENTER_X + 40} y2={TOP_Y + 445} color={COLORS.lightOrange} opacity={fadeIn(frame, FUSION_START + 15)} progress={fadeIn(frame, FUSION_START + 15)} />
            <Arrow x1={RIGHT_X + 40} y1={TOP_Y + 410} x2={CENTER_X - 90 + 260} y2={TOP_Y + 445} color={COLORS.lightOrange} opacity={fadeIn(frame, FUSION_START + 15)} progress={fadeIn(frame, FUSION_START + 15)} />
          </Sequence>

          {/* --- Output Logits --- */}
          <Sequence from={OUTPUT_START}>
            <Box
              x={CENTER_X - 120}
              y={TOP_Y + 550}
              w={320}
              h={80}
              color={COLORS.output}
              label="Final Logits"
              sublabel="w·s2v + (1-w)·v2s + local_score"
              borderColor={COLORS.lightRed}
              opacity={fadeIn(frame, OUTPUT_START)}
              scale={scaleIn(fps, OUTPUT_START)}
            />

            <Arrow x1={CENTER_X + 40} y1={TOP_Y + 480} x2={CENTER_X + 40} y2={TOP_Y + 547} color={COLORS.lightRed} opacity={fadeIn(frame, OUTPUT_START)} progress={1} />
          </Sequence>

          {/* --- GZSL Classification --- */}
          <Sequence from={OUTPUT_START + 30}>
            <div style={{
              position: "absolute",
              left: CENTER_X - 180, top: TOP_Y + 680,
              width: 480, height: 70,
              backgroundColor: COLORS.card,
              borderRadius: 12,
              border: "1px solid " + COLORS.cardBorder,
              opacity: fadeIn(frame, OUTPUT_START + 30),
              transform: `scale(${scaleIn(fps, OUTPUT_START + 30)})`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 30,
            }}>
              <span style={{ fontSize: 16, fontWeight: 700, color: COLORS.lightBlue }}>
                Seen (150) → S
              </span>
              <span style={{ fontSize: 16, fontWeight: 700, color: COLORS.lightGreen }}>
                Unseen (50) → U
              </span>
              <span style={{ fontSize: 18, fontWeight: 800, color: COLORS.lightRed }}>
                H = 2·U·S/(U+S)
              </span>
            </div>

            <Arrow x1={CENTER_X + 40} y1={TOP_Y + 630} x2={CENTER_X + 40} y2={TOP_Y + 677} color={COLORS.lightRed} opacity={fadeIn(frame, OUTPUT_START + 30)} progress={1} />
          </Sequence>

          {/* --- AG-JEPA --- */}
          <Sequence from={JEPA_START}>
            <div style={{
              position: "absolute",
              left: LEFT_X, top: TOP_Y + 510,
              width: 220, height: 120,
              backgroundColor: COLORS.card,
              borderRadius: 12,
              border: "1px solid " + COLORS.cardBorder,
              opacity: fadeIn(frame, JEPA_START),
              transform: `scale(${scaleIn(fps, JEPA_START)})`,
              display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center",
              padding: 8,
              gap: 4,
            }}>
              <span style={{ fontSize: 15, fontWeight: 700, color: COLORS.lightBlue }}>AG-JEPA (Aux)</span>
              <span style={{ fontSize: 11, color: COLORS.gray, textAlign: "center" }}>
                Top-K patch selection via
              </span>
              <span style={{ fontSize: 11, color: COLORS.gray, textAlign: "center" }}>
                class text similarity
              </span>
              <span style={{ fontSize: 10, color: COLORS.gray, textAlign: "center" }}>
                jepa_topk=8 · λ=0.05
              </span>
            </div>

            {/* Dashed arrow from patches to JEPA */}
            <div style={{
              position: "absolute",
              left: LEFT_X + 110, top: TOP_Y + 320,
              width: 2,
              height: TOP_Y + 507 - (TOP_Y + 320),
              background: `linear-gradient(to bottom, ${COLORS.lightBlue}, transparent 50%, ${COLORS.lightBlue})`,
              opacity: fadeIn(frame, JEPA_START),
            }} />
          </Sequence>

          {/* --- Loss Functions --- */}
          <Sequence from={LOSSES_START}>
            <div style={{
              position: "absolute",
              right: 80, top: TOP_Y + 370,
              width: 220, height: 170,
              backgroundColor: COLORS.card,
              borderRadius: 12,
              border: "1px solid " + COLORS.cardBorder,
              opacity: fadeIn(frame, LOSSES_START),
              transform: `scale(${scaleIn(fps, LOSSES_START)})`,
              display: "flex", flexDirection: "column",
              alignItems: "flex-start", justifyContent: "center",
              padding: 16,
              gap: 6,
            }}>
              <span style={{ fontSize: 14, fontWeight: 700, color: COLORS.lightOrange, marginBottom: 4 }}>
                Loss Functions
              </span>
              <span style={{ fontSize: 12, color: COLORS.gray }}>L_reg: 0.3</span>
              <span style={{ fontSize: 12, color: COLORS.gray }}>L_con: 0.2</span>
              <span style={{ fontSize: 12, color: COLORS.gray }}>L_topo: 0.05</span>
              <span style={{ fontSize: 12, color: COLORS.gray }}>L_jepa: 0.05</span>
              <span style={{ fontSize: 12, color: COLORS.gray }}>L_jepa_neg: 0.01</span>
              <span style={{ fontSize: 12, color: COLORS.gray }}>L_cond_text: 0.01</span>
              <span style={{ fontSize: 12, color: COLORS.gray }}>L_consist: 0.05</span>
              <span style={{ fontSize: 12, color: COLORS.gray }}>L_msdn: 0.05</span>
            </div>
          </Sequence>

          {/* --- Results Card --- */}
          <Sequence from={RESULT_START}>
            <div style={{
              position: "absolute",
              bottom: 30, left: "50%",
              transform: `translateX(-50%) scale(${scaleIn(fps, RESULT_START)})`,
              backgroundColor: COLORS.card,
              border: `1px solid ${COLORS.cardBorder}`,
              borderRadius: 16,
              padding: "16px 40px",
              opacity: fadeIn(frame, RESULT_START),
              display: "flex",
              gap: 40,
              alignItems: "center",
            }}>
              <div style={{ textAlign: "center" }}>
                <span style={{ fontSize: 32, fontWeight: 800, color: COLORS.lightBlue }}>72.33</span>
                <p style={{ fontSize: 13, color: COLORS.gray, margin: 0 }}>U (Unseen)</p>
              </div>
              <div style={{ textAlign: "center" }}>
                <span style={{ fontSize: 32, fontWeight: 800, color: COLORS.lightGreen }}>75.95</span>
                <p style={{ fontSize: 13, color: COLORS.gray, margin: 0 }}>S (Seen)</p>
              </div>
              <div style={{ textAlign: "center" }}>
                <span style={{ fontSize: 44, fontWeight: 900, color: COLORS.lightRed }}>74.09</span>
                <p style={{ fontSize: 14, color: COLORS.gray, margin: 0 }}>H (Harmonic Mean)</p>
              </div>
              <div style={{ textAlign: "center" }}>
                <span style={{ fontSize: 32, fontWeight: 800, color: COLORS.lightPurple }}>81.75</span>
                <p style={{ fontSize: 13, color: COLORS.gray, margin: 0 }}>ZS (Zero-Shot)</p>
              </div>
            </div>
          </Sequence>
        </>
      )}
    </AbsoluteFill>
  );
};

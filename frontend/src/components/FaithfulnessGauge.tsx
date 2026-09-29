interface FaithfulnessGaugeProps {
  limeScore: number
  shapScore: number
}

const RADIUS = 80
const ARC_LENGTH = Math.PI * RADIUS

export function FaithfulnessGauge({ limeScore, shapScore }: FaithfulnessGaugeProps) {
  const combined = (limeScore + shapScore) / 2
  const offset = ARC_LENGTH * (1 - Math.max(0, Math.min(1, combined)))

  return (
    <div className="gauge-card">
      <svg viewBox="0 0 200 110" width="200" height="110" role="img" aria-label="Faithfulness gauge">
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke="var(--border)"
          strokeWidth={14}
          strokeLinecap="round"
        />
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke="var(--teal)"
          strokeWidth={14}
          strokeLinecap="round"
          strokeDasharray={ARC_LENGTH}
          strokeDashoffset={offset}
        />
      </svg>
      <span className="gauge-value">{combined.toFixed(2)}</span>
      <span className="gauge-breakdown">
        LIME {limeScore.toFixed(2)} · SHAP {shapScore.toFixed(2)}
      </span>
    </div>
  )
}

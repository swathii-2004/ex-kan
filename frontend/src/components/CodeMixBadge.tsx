interface CodeMixBadgeProps {
  bucket: 'low' | 'medium' | 'high'
  effectiveScore: number
}

export function CodeMixBadge({ bucket, effectiveScore }: CodeMixBadgeProps) {
  return (
    <span className={`badge badge--${bucket}`}>
      {bucket} mix · <span className="mono">{effectiveScore.toFixed(2)}</span>
    </span>
  )
}

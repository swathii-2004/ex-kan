export function Footer() {
  return (
    <footer className="app-footer">
      <p>
        Research prototype. Predicts a sentiment-based proxy for distress, trained on
        YouTube comments, with only two classes (distress proxy / not). Neutral or
        non-sentiment text is assigned to one of them unreliably, and confidence does
        not indicate this. It is not a diagnostic or clinical tool.
      </p>
    </footer>
  )
}

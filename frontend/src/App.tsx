import { Navigate, Route, Routes } from 'react-router-dom'
import { TopNav } from './components/TopNav'
import { AnalysisProvider } from './context/AnalysisContext'
import { AnalyzeScreen } from './pages/AnalyzeScreen'
import { DashboardScreen } from './pages/DashboardScreen'
import { HistoryScreen } from './pages/HistoryScreen'
import { ResultScreen } from './pages/ResultScreen'

function App() {
  return (
    <AnalysisProvider>
      <TopNav />
      <Routes>
        <Route path="/" element={<AnalyzeScreen />} />
        <Route path="/result" element={<ResultScreen />} />
        <Route path="/dashboard" element={<DashboardScreen />} />
        <Route path="/history" element={<HistoryScreen />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AnalysisProvider>
  )
}

export default App

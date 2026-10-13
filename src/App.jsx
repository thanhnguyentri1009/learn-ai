import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { createContext, useContext, useState } from 'react'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import TrackPage from './pages/TrackPage'
import LessonPage from './pages/LessonPage'
import 'highlight.js/styles/atom-one-dark.css'

export const ThemeContext = createContext()
export const useTheme = () => useContext(ThemeContext)

export default function App() {
  const [theme, setTheme] = useState('dark')
  const toggle = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  document.body.setAttribute('data-theme', theme)

  return (
    <ThemeContext.Provider value={{ theme, toggle }}>
      <BrowserRouter>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/track/:trackId" element={<TrackPage />} />
          <Route path="/track/:trackId/lesson/:lessonId" element={<LessonPage />} />
        </Routes>
      </BrowserRouter>
    </ThemeContext.Provider>
  )
}

import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import { useTheme } from '../App'

function BackButton() {
  const location = useLocation()
  const navigate = useNavigate()

  // On lesson page: go back to track
  // On track page: go back to home
  const isOnLesson = location.pathname.includes('/lesson/')
  const isOnTrack = /^\/track\/[^/]+$/.test(location.pathname)

  if (!isOnLesson && !isOnTrack) return null

  const handleBack = () => {
    if (isOnLesson) {
      // Navigate back to the track page
      const parts = location.pathname.split('/lesson/')
      navigate(parts[0])
    } else {
      navigate('/')
    }
  }

  return (
    <button className="navbar-back" onClick={handleBack}>
      ← Quay lại
    </button>
  )
}

export default function Navbar() {
  const { theme, toggle } = useTheme()

  return (
    <nav className="navbar">
      <div className="navbar-left">
        <Link to="/" className="navbar-logo">
          LearnAI
        </Link>
        <BackButton />
      </div>
      <div className="navbar-right">
        <button
          className="theme-toggle"
          onClick={toggle}
          title={theme === 'dark' ? 'Chuyển sang sáng' : 'Chuyển sang tối'}
        >
          {theme === 'dark' ? '☀️' : '🌙'}
        </button>
      </div>
    </nav>
  )
}

import { useParams, Link } from 'react-router-dom'
import { track as aiTrack } from '../data/aiAppEngineer'
import { track as mlTrack } from '../data/mlEngineer'

const tracks = {
  'ai-app-engineer': aiTrack,
  'ml-engineer': mlTrack,
}

function difficultyClass(difficulty) {
  if (difficulty === 'Cơ bản') return 'badge badge-basic'
  if (difficulty === 'Trung cấp') return 'badge badge-mid'
  if (difficulty === 'Nâng cao') return 'badge badge-adv'
  return 'badge badge-basic'
}

export default function TrackPage() {
  const { trackId } = useParams()
  const track = tracks[trackId]

  if (!track) {
    return (
      <div className="track-page">
        <p>Không tìm thấy track này.</p>
      </div>
    )
  }

  return (
    <div className="track-page">
      <div className="track-header">
        <div className="track-header-icon">{track.icon}</div>
        <h1>{track.title}</h1>
        <p>{track.description}</p>
      </div>

      <div className="lesson-list">
        {track.lessons.map((lesson, index) => (
          <Link
            key={lesson.id}
            to={`/track/${trackId}/lesson/${lesson.id}`}
            className="lesson-card"
          >
            <div className="lesson-number">
              {String(index + 1).padStart(2, '0')}
            </div>
            <div className="lesson-info">
              <h3>{lesson.title}</h3>
              <p>{lesson.description}</p>
            </div>
            <span className={difficultyClass(lesson.difficulty)}>
              {lesson.difficulty}
            </span>
          </Link>
        ))}
      </div>
    </div>
  )
}

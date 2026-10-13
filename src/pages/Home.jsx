import { Link } from 'react-router-dom'

const tracks = [
  {
    id: 'ai-app-engineer',
    icon: '🤖',
    title: 'AI Application Engineer',
    description: 'Xây dựng ứng dụng AI sử dụng LLM APIs, RAG, và các framework hiện đại. Phù hợp cho developer muốn tích hợp AI vào sản phẩm thực tế.',
    tags: ['LLM APIs', 'Prompt Engineering', 'RAG', 'LangChain', 'Vector DB'],
    lessons: 5,
  },
  {
    id: 'ml-engineer',
    icon: '🧠',
    title: 'Machine Learning Engineer',
    description: 'Nắm vững thuật toán ML, deep learning, và deploy model vào production. Từ Python cơ bản đến PyTorch và MLOps.',
    tags: ['Python/NumPy', 'Scikit-learn', 'Neural Networks', 'PyTorch', 'MLOps'],
    lessons: 5,
  },
]

export default function Home() {
  return (
    <div className="home">
      <div className="hero">
        <h1>Master AI Engineering</h1>
        <p>Học AI từ cơ bản đến nâng cao để sẵn sàng cho công việc</p>
      </div>

      <div className="cards-grid">
        {tracks.map(track => (
          <div key={track.id} className="track-card">
            <div className="track-card-icon">{track.icon}</div>
            <h2>{track.title}</h2>
            <p>{track.description}</p>
            <div className="track-tags">
              {track.tags.map(tag => (
                <span key={tag} className="tag">{tag}</span>
              ))}
            </div>
            <Link to={`/track/${track.id}`} className="track-card-btn">
              Bắt đầu học →
            </Link>
          </div>
        ))}
      </div>
    </div>
  )
}

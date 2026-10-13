import { useParams } from 'react-router-dom'
import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'

// Eagerly load all markdown files as raw text
const modules = import.meta.glob('../content/**/*.md', { query: '?raw', import: 'default', eager: true })

const mdOptions = {
  remarkPlugins: [remarkGfm],
  rehypePlugins: [rehypeHighlight],
}

function MarkdownSection({ content }) {
  return (
    <div className="markdown-content">
      <ReactMarkdown {...mdOptions}>{content}</ReactMarkdown>
    </div>
  )
}

export default function LessonPage() {
  const { trackId, lessonId } = useParams()
  const [showSolution, setShowSolution] = useState(false)

  const key = `../content/${trackId}/${lessonId}.md`
  const raw = modules[key]

  if (!raw) {
    return (
      <div className="lesson-page">
        <div className="loading">Không tìm thấy bài học này.</div>
      </div>
    )
  }

  // Split content by special delimiters
  const PRACTICE_DELIMITER = '---PRACTICE---'
  const SOLUTION_DELIMITER = '---SOLUTION---'

  const practiceIndex = raw.indexOf(PRACTICE_DELIMITER)
  const solutionIndex = raw.indexOf(SOLUTION_DELIMITER)

  let theoryContent = raw
  let practiceContent = ''
  let solutionContent = ''

  if (practiceIndex !== -1) {
    theoryContent = raw.slice(0, practiceIndex).trim()
    if (solutionIndex !== -1) {
      practiceContent = raw.slice(practiceIndex + PRACTICE_DELIMITER.length, solutionIndex).trim()
      solutionContent = raw.slice(solutionIndex + SOLUTION_DELIMITER.length).trim()
    } else {
      practiceContent = raw.slice(practiceIndex + PRACTICE_DELIMITER.length).trim()
    }
  }

  return (
    <div className="lesson-page">
      <div className="lesson-sections">

        {/* Theory Section */}
        <div className="section-card">
          <div className="section-header">
            <span className="section-header-icon">📖</span>
            <h2>Lý thuyết</h2>
          </div>
          <div className="section-body">
            <MarkdownSection content={theoryContent} />
          </div>
        </div>

        {/* Practice Section */}
        {practiceContent && (
          <div className="section-card">
            <div className="section-header">
              <span className="section-header-icon">💻</span>
              <h2>Bài tập thực hành</h2>
            </div>
            <div className="section-body">
              <MarkdownSection content={practiceContent} />

              {solutionContent && (
                <>
                  <button
                    className="solution-toggle"
                    onClick={() => setShowSolution(s => !s)}
                  >
                    {showSolution ? '🙈 Ẩn lời giải' : '💡 Xem lời giải'}
                  </button>

                  {showSolution && (
                    <div className="solution-box">
                      <MarkdownSection content={solutionContent} />
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}

      </div>
    </div>
  )
}


import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Send, LogOut, CheckCircle, AlertTriangle, FileText, Image as ImageIcon } from 'lucide-react'

interface Props {
    token: string
    setToken: (token: string | null) => void
}

interface Message {
    role: 'user' | 'assistant'
    content: string
    files?: Record<string, string>
    diagram_b64?: string
    validation_status?: string
}

const Chat = ({ token, setToken }: Props) => {
    const [messages, setMessages] = useState<Message[]>([])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [conversations, setConversations] = useState<any[]>([])
    const [activeConvId, setActiveConvId] = useState<number | null>(null)
    const bottomRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        fetchConversations()
    }, [])

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    const fetchConversations = async () => {
        try {
            const res = await axios.get('/api/conversations', { headers: { Authorization: `Bearer ${token}` } })
            setConversations(res.data)
        } catch (e) { console.error(e) }
    }

    const loadConversation = async (id: number) => {
        setActiveConvId(id)
        try {
            const res = await axios.get(`/api/conversations/${id}`, { headers: { Authorization: `Bearer ${token}` } })
            setMessages(res.data)
        } catch (e) { console.error(e) }
    }

    const sendMessage = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!input.trim()) return

        const userMsg: Message = { role: 'user', content: input }
        setMessages(prev => [...prev, userMsg])
        setInput('')
        setLoading(true)

        try {
            const res = await axios.post('/api/chat',
                { message: userMsg.content, conversation_id: activeConvId },
                { headers: { Authorization: `Bearer ${token}` } }
            )

            const aiMsg: Message = {
                role: 'assistant',
                content: res.data.response,
                files: res.data.files,
                diagram_b64: res.data.diagram_b64,
                validation_status: res.data.validation_status
            }

            setMessages(prev => [...prev, aiMsg])
            if (!activeConvId) {
                setActiveConvId(res.data.conversation_id)
                fetchConversations()
            }
        } catch (err) {
            console.error(err)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="flex h-screen overflow-hidden">
            {/* Sidebar */}
            <div className="w-64 bg-gray-800 border-r border-gray-700 flex-shrink-0 flex flex-col">
                <div className="p-4 border-b border-gray-700 flex justify-between items-center">
                    <h1 className="font-bold text-lg text-blue-400">Architect AI</h1>
                    <button onClick={() => setToken(null)}><LogOut size={18} /></button>
                </div>
                <div className="flex-1 overflow-y-auto p-2">
                    <button
                        onClick={() => { setActiveConvId(null); setMessages([]); }}
                        className="w-full text-left p-3 rounded hover:bg-gray-700 mb-2 border border-dashed border-gray-600 text-sm text-gray-400"
                    >
                        + New Chat
                    </button>
                    {conversations.map(c => (
                        <button
                            key={c.id}
                            onClick={() => loadConversation(c.id)}
                            className={`w-full text-left p-3 rounded mb-1 text-sm truncate ${activeConvId === c.id ? 'bg-blue-900/50 text-blue-200' : 'hover:bg-gray-700'}`}
                        >
                            {c.title}
                        </button>
                    ))}
                </div>
            </div>

            {/* Main Chat */}
            <div className="flex-1 flex flex-col bg-gray-900 relative">
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {!activeConvId && messages.length === 0 && (
                        <div className="flex h-full items-center justify-center text-gray-500 flex-col">
                            <div className="text-6xl mb-4">🧠</div>
                            <h2 className="text-2xl font-bold mb-2">Terraform Architect Agent</h2>
                            <p>Describe your infrastructure, and we'll build, validate, and visualize it.</p>
                        </div>
                    )}

                    {messages.map((msg, i) => (
                        <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-4xl p-4 rounded-lg ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-800 border border-gray-700'}`}>
                                <ReactMarkdown className="markdown-body text-sm space-y-2">
                                    {msg.content}
                                </ReactMarkdown>

                                {/* Files Tab */}
                                {msg.files && Object.keys(msg.files).length > 0 && (
                                    <div className="mt-4 bg-gray-950 rounded border border-gray-800 overflow-hidden">
                                        <div className="bg-gray-900 px-3 py-1 text-xs border-b border-gray-800 flex gap-2">
                                            <FileText size={14} className="text-blue-400" /> Generated Files
                                        </div>
                                        <div className="p-0">
                                            {Object.entries(msg.files).map(([fname, code]) => (
                                                <div key={fname} className="mb-0">
                                                    <div className="px-4 py-1 bg-gray-800 text-xs text-gray-400 font-mono border-b border-gray-700">{fname}</div>
                                                    <SyntaxHighlighter language="hcl" style={vscDarkPlus} customStyle={{ margin: 0, borderRadius: 0 }}>
                                                        {code}
                                                    </SyntaxHighlighter>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Validation Status */}
                                {msg.validation_status && msg.validation_status !== "Skipped" && (
                                    <div className={`mt-2 text-xs flex items-center gap-1 ${msg.validation_status === 'Success' ? 'text-green-400' : 'text-amber-400'}`}>
                                        {msg.validation_status === 'Success' ? <CheckCircle size={14} /> : <AlertTriangle size={14} />}
                                        Validation: {msg.validation_status}
                                    </div>
                                )}

                                {/* Diagram */}
                                {msg.diagram_b64 && (
                                    <div className="mt-4">
                                        <div className="flex items-center gap-2 text-xs text-purple-400 mb-1">
                                            <ImageIcon size={14} /> Architecture Diagram
                                        </div>
                                        <img
                                            src={`data:image/png;base64,${msg.diagram_b64}`}
                                            alt="Architecture Diagram"
                                            className="rounded border border-gray-700 max-w-full hover:scale-105 transition-transform cursor-pointer bg-white"
                                        />
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                    {loading && (
                        <div className="flex justify-start">
                            <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 animate-pulse text-sm text-gray-400">
                                🧠 Architecting solution... (Validating & Visualizing)
                            </div>
                        </div>
                    )}
                    <div ref={bottomRef} />
                </div>

                {/* Input */}
                <div className="p-4 border-t border-gray-800 bg-gray-900">
                    <form onSubmit={sendMessage} className="relative max-w-4xl mx-auto">
                        <input
                            type="text"
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            placeholder="E.g., Create a scalable ECS cluster with Application Load Balancer..."
                            className="w-full bg-gray-800 text-white rounded-full py-3 px-6 pr-12 focus:outline-none focus:ring-2 focus:ring-blue-600 shadow-lg border border-gray-700"
                            disabled={loading}
                        />
                        <button
                            type="submit"
                            disabled={loading || !input.trim()}
                            className="absolute right-2 top-2 p-2 bg-blue-600 rounded-full text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <Send size={18} />
                        </button>
                    </form>
                </div>
            </div>
        </div>
    )
}

export default Chat

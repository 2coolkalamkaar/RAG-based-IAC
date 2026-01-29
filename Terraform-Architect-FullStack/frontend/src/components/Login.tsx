
import { useState } from 'react'
import axios from 'axios'
import { useNavigate } from 'react-router-dom'
import { Lock, User } from 'lucide-react'

interface Props {
    setToken: (token: string) => void
}

const Login = ({ setToken }: Props) => {
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const [isRegistering, setIsRegistering] = useState(false)
    const [error, setError] = useState('')
    const navigate = useNavigate()

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setError('')
        const endpoint = isRegistering ? '/api/auth/register' : '/api/auth/token'

        try {
            const res = await axios.post(endpoint, { username, password })
            setToken(res.data.access_token)
            navigate('/')
        } catch (err: any) {
            setError(err.response?.data?.detail || 'An error occurred')
        }
    }

    return (
        <div className="flex items-center justify-center min-h-screen">
            <div className="w-full max-w-md p-8 space-y-8 bg-gray-800 rounded-lg shadow-xl">
                <h2 className="text-3xl font-bold text-center text-blue-500">
                    {isRegistering ? 'Create Account' : 'Welcome Back'}
                </h2>

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="relative">
                        <User className="absolute left-3 top-3 text-gray-400" size={20} />
                        <input
                            type="text"
                            placeholder="Username"
                            className="w-full py-2 pl-10 pr-4 bg-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                        />
                    </div>
                    <div className="relative">
                        <Lock className="absolute left-3 top-3 text-gray-400" size={20} />
                        <input
                            type="password"
                            placeholder="Password"
                            className="w-full py-2 pl-10 pr-4 bg-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                        />
                    </div>

                    {error && <p className="text-red-500 text-sm">{error}</p>}

                    <button
                        type="submit"
                        className="w-full py-2 font-bold text-white bg-blue-600 rounded hover:bg-blue-700 transition"
                    >
                        {isRegistering ? 'Register' : 'Login'}
                    </button>
                </form>

                <p className="text-center text-gray-400">
                    {isRegistering ? 'Already have an account?' : "Don't have an account?"}{' '}
                    <button
                        onClick={() => setIsRegistering(!isRegistering)}
                        className="text-blue-400 hover:underline"
                    >
                        {isRegistering ? 'Login' : 'Register'}
                    </button>
                </p>
            </div>
        </div>
    )
}

export default Login


import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import Chat from './components/Chat'
import Login from './components/Login'

function App() {
    const [token, setToken] = useState<string | null>(localStorage.getItem('token'))

    useEffect(() => {
        if (token) {
            localStorage.setItem('token', token)
        } else {
            localStorage.removeItem('token')
        }
    }, [token])

    return (
        <Router>
            <div className="min-h-screen bg-gray-900 text-white">
                <Routes>
                    <Route path="/login" element={<Login setToken={setToken} />} />
                    <Route
                        path="/"
                        element={token ? <Chat token={token} setToken={setToken} /> : <Navigate to="/login" />}
                    />
                </Routes>
            </div>
        </Router>
    )
}

export default App

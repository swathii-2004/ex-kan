import { NavLink } from 'react-router-dom'

const LINKS = [
  { to: '/', label: 'Analyze' },
  { to: '/result', label: 'Result' },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/history', label: 'History' },
]

export function TopNav() {
  return (
    <header className="top-nav">
      <NavLink to="/" className="top-nav__brand">
        Ex-KAN
      </NavLink>
      <nav>
        <ul className="top-nav__links">
          {LINKS.map((link) => (
            <li key={link.to}>
              <NavLink
                to={link.to}
                end={link.to === '/'}
                className={({ isActive }) =>
                  isActive ? 'top-nav__link top-nav__link--active' : 'top-nav__link'
                }
              >
                {link.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  )
}

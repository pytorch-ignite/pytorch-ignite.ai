function activeNavBar() {
  const currentURL = location.pathname
  const navbarMenu = document.getElementById('navbar-menu')
  const navbarItems = navbarMenu.querySelectorAll('a.navbar-item')
  for (const navbarItem of navbarItems) {
    const href = navbarItem.getAttribute('href')
    if (href && currentURL.search(href) > -1) {
      navbarItem.classList.add('active')
    }
  }
}

activeNavBar()

function toggle() {
  const sun = document.getElementById('sun')
  const moon = document.getElementById('moon')
  const toggle = document.getElementById('toggleDark')

  if (document.documentElement.classList.contains('dark')) {
    // dark mode has turned on, change sun to moon
    toggleClassList()
  }

  save()

  toggle.addEventListener('click', function (e) {
    toggleClassList()
    document.documentElement.classList.toggle('dark')
    save()
  })

  function toggleClassList() {
    sun.classList.toggle('hidden')
    moon.classList.toggle('hidden')
  }

  function save() {
    const value = sun.classList.contains('hidden') ? 'dark' : 'light'
    localStorage.setItem('ignite-color-scheme', value)
  }
}

toggle()

document.querySelectorAll('h2, h3').forEach((value) => {
  const child = document.createElement('a')
  child.href = '#' + value.id
  child.className = 'header-anchor'
  child.innerHTML = `<span class="sr-only">#${value.id}</span>`
  child.setAttribute('aria-hidden', true)
  value.prepend(child)
})

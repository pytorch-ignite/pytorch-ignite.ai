async function highLightWithShiki() {
  const highligher = await shiki.getHighlighter({
    theme: 'github-dark',
    langs: ['py', 'shell']
  })

  const preBlocks = document.querySelectorAll('pre[style]')

  for (const block of preBlocks) {
    const output = highligher.codeToHtml(block.textContent, block.firstElementChild.getAttribute('data-lang') || 'text')
    block.outerHTML = output
  }
}

highLightWithShiki()

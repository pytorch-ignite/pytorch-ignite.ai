// @ts-check
// only used in development
async function highLightWithShiki() {
  const highligher = await shiki.getHighlighter({
    theme: 'github-dark',
    langs: ['py', 'shell']
  })

  const preBlocks = document.querySelectorAll('pre[style]')

  for (const block of preBlocks) {
    const html = highligher.codeToHtml(block.textContent, block.firstElementChild.getAttribute('data-lang') || 'text')
    block.outerHTML = html
  }
}

highLightWithShiki()

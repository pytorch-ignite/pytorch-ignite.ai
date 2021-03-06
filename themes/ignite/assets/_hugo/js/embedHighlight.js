// @ts-check
// only used in development

export async function highLightWithShiki(shiki, document = window.document) {
  const highligher = await shiki.getHighlighter({
    theme: 'one-dark-pro',
    langs: ['py', 'shell'],
  })

  const preBlocks = document.querySelectorAll('pre[style]')

  for (const block of preBlocks) {
    const html = highligher.codeToHtml(
      block.textContent,
      block.firstElementChild.getAttribute('data-lang') || 'text'
    )
    block.outerHTML = html
  }
}

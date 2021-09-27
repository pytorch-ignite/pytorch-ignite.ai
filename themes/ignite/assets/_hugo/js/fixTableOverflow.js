// @ts-check

export function fixTableOverflow(document = window.document) {
  const tables = document.querySelectorAll('table.dataframe')

  if (tables) {
    for (const table of tables) {
      const div = table.closest('div')
      div.style.overflow = 'auto'
    }
  }
}

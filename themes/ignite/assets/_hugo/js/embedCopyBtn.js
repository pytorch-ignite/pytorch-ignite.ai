// reference from
// https://www.dannyguo.com/blog/how-to-add-copy-to-clipboard-buttons-to-code-blocks-in-hugo/

function embedCopyBtn() {
  document.querySelectorAll('pre[style] > code').forEach((v) => {
    const button = document.createElement('button')
    button.id = 'copyBtn'
    button.className = 'absolute top-1 right-1'
    button.innerHTML = `
    <svg id="copyIt" class="h-6 w-6" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" width="32" height="32" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M28 10v18H10V10h18m0-2H10a2 2 0 0 0-2 2v18a2 2 0 0 0 2 2h18a2 2 0 0 0 2-2V10a2 2 0 0 0-2-2z" fill="currentColor"></path><path d="M4 18H2V4a2 2 0 0 1 2-2h14v2H4z" fill="currentColor"></path></svg>
    <svg id="copyDone" xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 hidden" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
    `
    v.closest('.highlight').appendChild(button)
  })
}
embedCopyBtn()

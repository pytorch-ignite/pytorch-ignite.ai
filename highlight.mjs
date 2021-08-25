// @ts-check

// We are doing pre-highlighting with Shiki of the built html files
// so that we don't need to ship Shiki on client side.
// In development, we are using Shiki from jsdelivr so
// normal `hugo server` just works.
// This is used in Netlify build.

import * as fs from 'fs'
import * as path from 'path'
import { JSDOM } from 'jsdom'
import { getHighlighter } from 'shiki'

function* walkDir(dir) {
  const files = fs.readdirSync(dir, { withFileTypes: true })
  for (const file of files) {
    if (file.isDirectory()) {
      yield* walkDir(path.join(dir, file.name))
    } else if (file.isFile() && file.name.endsWith('.html')) {
      yield path.join(dir, file.name)
    }
  }
}

async function preHighLightWithShiki(path) {
  try {
    for (const file of walkDir(path)) {
      const html = fs.readFileSync(file, { encoding: 'utf-8' })
      const { document } = new JSDOM(html, { contentType: 'text/html' }).window

      // same code as in shiki.js
      const highligher = await getHighlighter({
        theme: 'github-dark',
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

      // update with Shiki highlighted html
      fs.writeFileSync(
        file,
        '<!DOCTYPE html>' + document.documentElement.outerHTML,
        { encoding: 'utf-8' }
      )
      console.log(`Updated ${file}`)
    }
  } catch (e) {
    throw e
  }
}

preHighLightWithShiki('dist')

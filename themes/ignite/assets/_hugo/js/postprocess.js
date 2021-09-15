// @ts-check

// We are doing
// 1. pre-highlighting with Shiki of the built html files
// so that we don't need to ship Shiki on client side.
// In development, we are using Shiki from jsdelivr so
// normal `hugo server` just works.
// 2. We are embedding copy button to the code blocks
// since it can be built statically before hand.
// 3. We are also fixing long table overflow.
// This is used in Netlify build.

import * as fs from 'fs'
import * as path from 'path'
import { JSDOM } from 'jsdom'
import shiki from 'shiki'
import { highLightWithShiki } from './embedHighlight.js'
import { embedCopyBtn } from './embedCopyBtn.js'
import { fixTableOverflow } from './fixTableOverflow.js'

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

async function postprocess(path) {
  try {
    for (const file of walkDir(path)) {
      const html = fs.readFileSync(file, { encoding: 'utf-8' })
      const { document, location } = new JSDOM(html, {
        contentType: 'text/html',
      }).window

      await highLightWithShiki(shiki, document)
      if (file !== 'dist/index.html') {
        embedCopyBtn(document, location)
      }
      fixTableOverflow(document)

      // update the built html files
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

postprocess('dist')

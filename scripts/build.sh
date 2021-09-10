set -e

npm i pnpm -g
pnpm i --frozen-lockfile --store=node_modules/.pnpm-store
pnpm build
hugo --gc --minify
node highlight.mjs

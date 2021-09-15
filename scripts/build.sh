set -e

npm i pnpm -g
pnpm i --frozen-lockfile --store=node_modules/.pnpm-store
pnpm build
hugo --gc --minify
node themes/ignite/assets/_hugo/js/postprocess.js

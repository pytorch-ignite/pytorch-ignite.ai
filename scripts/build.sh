set -e

pnpm i --frozen-lockfile
pnpm build
hugo --gc --minify
node themes/ignite/assets/_hugo/js/postprocess.js

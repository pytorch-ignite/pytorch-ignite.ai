const { defineConfig } = require('windicss/helpers')
const typography = require('windicss/plugin/typography')
const colors = require('windicss/colors')

module.exports = defineConfig({
  darkMode: 'class',
  preflight: { includeAll: true },
  extract: { include: ['themes/**/*.{html,js}'] },
  plugins: [typography({ dark: true })],
  theme: {
    extend: {
      fontFamily: {
        sans: 'Inter var, Inter, ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji"',
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: 'none',
            a: {
              textDecoration: 'none',
              '&:hover': { textDecoration: 'underline' },
            },
            code: { color: colors.violet[500] },
          },
        },
      },
    },
  },
})

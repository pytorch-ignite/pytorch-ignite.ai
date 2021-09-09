const { defineConfig } = require("windicss/helpers")


module.exports = defineConfig({
  extract: {
    include: ['layouts/**/*.html']
  },
  theme: {
    extend: {
      fontFamily: {
        sans: 'Inter var, Inter, ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji"'
      }
    }
  }
})

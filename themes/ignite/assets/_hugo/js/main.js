function openSideBar() {
  const sideBarBtn = document.getElementById('sideBarBtn')
  const sidebar = document.getElementById('sidebar')

  sideBarBtn.addEventListener('click', function () {
    document.documentElement.classList.toggle('overflow-hidden')
    sidebar.classList.toggle('translate-x-0')
  })
}

openSideBar()

function openDropDown() {
  const matches = window.matchMedia('(max-width: 1024px)').matches

  if (matches) {
    const dropDownBtn = document.querySelector('#sidebar #dropDownBtn')
    const dropDownList = document.querySelector('#sidebar #dropDownList')
    const chevronRight = document.querySelector('#sidebar #chevronRight')
    chevronRight.classList.toggle('rotate-90')

    dropDownBtn.addEventListener('click', function () {
      dropDownList.classList.toggle('!block')
      chevronRight.classList.toggle('rotate-90')
    })
  }
}

openDropDown()
anchorScroll()

function anchorScroll() {
  window.addEventListener('click', function (e) {
    const link = e.target.closest('a')

    if (link) {
      const { protocol, hostname, pathname, hash, target } = link
      const currentUrl = window.location
      const extMatch = pathname.match(/\.\w+$/)

      if (
        !e.ctrlKey &&
        !e.shiftKey &&
        !e.altKey &&
        !e.metaKey &&
        target !== '_blank' &&
        protocol === currentUrl.protocol &&
        hostname === currentUrl.hostname &&
        !(extMatch && extMatch[0] !== '.html')
      ) {
        if (pathname === currentUrl.pathname) {
          e.preventDefault()
          if (hash && hash !== currentUrl.hash) {
            history.pushState(null, '', hash)
            const smooth = link.classList.contains('header-anchor')
            const target = link.classList.contains('header-anchor')
              ? link
              : document.querySelector(decodeURIComponent(hash))
            if (target) {
              const targetTop = target.offsetTop
              console.log(targetTop, window.scrollY, window.innerHeight)
              // only smooth scroll if distance is smaller than screen height.
              if (
                !smooth ||
                Math.abs(targetTop - window.scrollY) > window.innerHeight
              ) {
                window.scrollTo(0, targetTop)
              } else {
                window.scrollTo({
                  left: 0,
                  top: targetTop,
                  behavior: 'smooth',
                })
              }
            }
          }
        }
      }
    }
  })
}

if (location.pathname === '/') {
  const featuredPost = document.getElementById('featured-post-link')
  if (featuredPost.getAttribute('href') === 'https://github.com/pytorch/ignite/releases/latest') {
    fetch('https://api.github.com/repos/pytorch/ignite/releases/latest')
      .then(val => val.json())
      .then(val => {
        featuredPost.innerText = featuredPost.innerText + ' ' + val.tag_name
      })
  }
}

window.addEventListener('click', function (e) {
  if (['copyBtn', 'copyIt', 'copyDone'].includes(e.target.id)) {
    e.preventDefault()
    const copyIt = e.target
    const copyDone = copyIt.nextElementSibling
    const copyBtn = copyIt.closest('#copyBtn')
    const code = copyBtn.previousElementSibling

    navigator.clipboard.writeText(code.innerText).then(function () {
      e.target.blur()
      copyDone.classList.toggle('hidden')
      copyIt.classList.toggle('hidden')
      setTimeout(() => {
        copyDone.classList.toggle('hidden')
        copyIt.classList.toggle('hidden')
      }, 1500);
    }).catch(function (reason) {
      console.error(`${reason} Error copying code.`);
    })
  }
})

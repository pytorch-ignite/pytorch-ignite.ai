const pathname = location.pathname

function openSideBar() {
  if (pathname !== '/') {
    const sideBarBtn = document.getElementById('sideBarBtn')
    const sidebar = document.getElementById('sidebar')

    sideBarBtn.addEventListener('click', function () {
      sidebar.classList.toggle('translate-x-0')
    })
  }
}

function openDropDowns() {
  if (pathname !== '/') {
    const matches = window.matchMedia('(max-width: 1024px)').matches

    if (matches) {
      const chevronRights = document.querySelectorAll('#sidebar #chevronRight')
      for (const chevronRight of chevronRights) {
        chevronRight.classList.toggle('rotate-90')
      }

      window.addEventListener('click', function (e) {
        if (e.target.className.includes && e.target.className.includes('dropDownBtn')) {
          const dropDownBtn = e.target
          const chevronRight = dropDownBtn.nextElementSibling
          const dropDownList = chevronRight.nextElementSibling

          dropDownList.classList.toggle('!block')
          chevronRight.classList.toggle('rotate-90')
        }
      })
    }
  }
}

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
            const target = document.querySelector(decodeURIComponent(hash))
            if (target) {
              const targetTop = target.offsetTop
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
  })
}

function fetchRelease() {
  if (pathname === '/') {
    const featuredPost = document.getElementById('featured-post-link')
    if (
      featuredPost.getAttribute('href') ===
      'https://github.com/pytorch/ignite/releases/latest'
    ) {
      fetch('https://api.github.com/repos/pytorch/ignite/releases/latest')
        .then((val) => val.json())
        .then((val) => {
          featuredPost.innerText = featuredPost.innerText + ' ' + val.tag_name
        })
    }
  }
}

function copyCode() {
  window.addEventListener('click', function (e) {
    if (['copyIt', 'copyDone'].includes(e.target.id)) {
      e.preventDefault()
      const copyIt = e.target
      const copyDone = copyIt.nextElementSibling
      const copyBtn = copyIt.closest('.copyBtn')
      const code = copyBtn.previousElementSibling

      navigator.clipboard
        .writeText(code.innerText)
        .then(function () {
          e.target.blur()
          copyDone.classList.toggle('hidden')
          copyIt.classList.toggle('hidden')
          setTimeout(() => {
            copyDone.classList.toggle('hidden')
            copyIt.classList.toggle('hidden')
          }, 1500)
        })
        .catch(function (reason) {
          console.error(`${reason} Error copying code.`)
        })
    }
  })
}


openSideBar()
openDropDowns()
anchorScroll()
fetchRelease()
copyCode()

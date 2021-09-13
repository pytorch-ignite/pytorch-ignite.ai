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

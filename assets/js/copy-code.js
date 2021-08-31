// reference from
// https://www.dannyguo.com/blog/how-to-add-copy-to-clipboard-buttons-to-code-blocks-in-hugo/
function copyCode() {
  document.querySelectorAll(".highlight td:last-child pre > code").forEach(function (value) {
    const button = document.createElement("button");
    button.className = "copy-code-button";
    button.type = "button";
    button.innerText = "Copy";

    button.addEventListener("click", function (event) {
      event.preventDefault();

      navigator.clipboard
        .writeText(value.innerText)
        .then(function () {
          button.blur();
          button.innerText = "Copied";
          setTimeout(function () {
            button.innerText = "Copy";
          }, 1500);
        })
        .catch(function (reason) {
          console.error(`${reason} Error copying code.`);
        });
    });

    value.closest(".highlight").appendChild(button)
  });
}
copyCode();

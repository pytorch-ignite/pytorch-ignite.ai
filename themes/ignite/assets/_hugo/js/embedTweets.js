// @ts-check

async function embedTweet(iframe) {
  const tweets = []

  iframe.contentWindow.document
    .querySelectorAll('ol.timeline-TweetList > li')
    .forEach(function (t) {
      const isRetweet = t.querySelector('.timeline-Tweet-retweetCredit')
      const tweetAuthor = t.querySelector('.tweetAuthor-screenName').innerText
      let inReplyTo = t.querySelector('.timeline-Tweet-inReplyTo')
      if (inReplyTo) {
        inReplyTo = inReplyTo.innerText
      }
      const tweetHTML = t.querySelector('p.timeline-tweet-text').innerHTML

      tweets.push({
        isRetweet,
        tweetAuthor,
        inReplyTo,
        tweetHTML,
      })
    })

  let tweetsWrapper = `<div class="grid gap-x-4 gap-y-6 prose prose-red sm:grid-cols-3">`

  for (const tweet of tweets) {
    let tweetWrapper = `<div class="border rounded-lg p-4 dark:border-dark-200">
    <svg class="h-6 w-6" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true"
    role="img" width="32" height="32" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24">
    <path
      d="M22.46 6c-.77.35-1.6.58-2.46.69c.88-.53 1.56-1.37 1.88-2.38c-.83.5-1.75.85-2.72 1.05C18.37 4.5 17.26 4 16 4c-2.35 0-4.27 1.92-4.27 4.29c0 .34.04.67.11.98C8.28 9.09 5.11 7.38 3 4.79c-.37.63-.58 1.37-.58 2.15c0 1.49.75 2.81 1.91 3.56c-.71 0-1.37-.2-1.95-.5v.03c0 2.08 1.48 3.82 3.44 4.21a4.22 4.22 0 0 1-1.93.07a4.28 4.28 0 0 0 4 2.98a8.521 8.521 0 0 1-5.33 1.84c-.34 0-.68-.02-1.02-.06C3.44 20.29 5.7 21 8.12 21C16 21 20.33 14.46 20.33 8.79c0-.19 0-.37-.01-.56c.84-.6 1.56-1.36 2.14-2.23z"
      fill="currentColor"></path>
    </svg>
    `

    let metadata = `<p>`

    if (tweet.isRetweet) {
      metadata += `<span>PyTorch-Ignite Retweeted ${tweet.tweetAuthor}<span><br />`
    }

    if (tweet.inReplyTo) {
      metadata += `<span>${tweet.inReplyTo}</span>`
    }

    tweetWrapper += metadata + '</p>'
    tweetWrapper += `<p>${tweet.tweetHTML}</p>`
    tweetWrapper += `
    <div>
      <a href="https://twitter.com/pytorch_ignite" target="_blank" rel="noopener noreferrer">
        @pytorch_ignite
      </a>
    </div>
    `

    tweetWrapper += `</div>`

    tweetsWrapper += tweetWrapper
  }

  document.querySelector("div[data-target='twitter-timeline']").innerHTML =
    tweetsWrapper + `</div>`
}

let count = 0
const interval = setInterval(function () {
  const tweets = document.getElementById('twitter-tweets')
  const iframe = document.getElementById('twitter-widget-0')
  if (iframe !== null) {
    clearInterval(interval)
    embedTweet(iframe)
  }

  // cancel after 5 seconds
  if (count > 5) {
    clearInterval(interval)
    tweets.innerText = 'Twitter widget could not be loaded.'
  } else {
    count += 1
  }
}, 1000)

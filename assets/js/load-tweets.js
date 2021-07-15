// @ts-check
async function embedTweet(iframe) {
  const tweets = $(iframe.contentWindow.document)
    .find("ol.timeline-TweetList > li")
    .map(function () {
      return {
        isRetweet: $(this).find(".timeline-Tweet-retweetCredit").length > 0,
        tweetAuthor: $(this).find(".tweetAuthor-screenName").text(),
        inReplyTo: $(this).find(".timeline-Tweet-inReplyTo").text(),
        tweetHTML: $(this).find("p.timeline-tweet-text").html(),
      };
    })
    .get();

  const tweetsWrapper = $('<div class="row tweets-wrapper"></div>');

  tweets.forEach(function (tweet) {
    const tweetWrapper = $('<div class="col-md-4 tweet"></div>');
    const metadata = $('<p class="tweet-header"></p>');

    if (tweet.isRetweet) {
      metadata.append(
        '<span class="retweeted">PyTorch-Ignite Retweeted ' +
          tweet.tweetAuthor +
          "</span><br />"
      );
    }

    if (tweet.inReplyTo) {
      metadata.append(
        '<span class="in-reply-to">' + tweet.inReplyTo + "</span>"
      );
    }

    tweetWrapper.append(metadata);

    tweetWrapper.append("<p>" + tweet.tweetHTML + "</p>");

    tweetWrapper.append(
      '<div class="tweet-author">\
        <a href="https://twitter.com/pytorch_ignite" target="_blank" class="twitter-handle">@pytorch_ignite</a> \
      </div>'
    );

    tweetWrapper.prepend(
      '<span class="icon"><i class="fa fa-twitter"></i></span>'
    );

    tweetsWrapper.append(tweetWrapper);
  });

  $("[data-target='twitter-timeline']").append(tweetsWrapper);
}

let count = 0;
const interval = setInterval(function () {
  const iframe = document.getElementById("twitter-widget-0");
  if (iframe !== null) {
    clearInterval(interval);
    embedTweet(iframe);
  }

  // cancel after 8 seconds
  if (count > 8) {
    clearInterval(interval);
    document.getElementById("twitter-tweets").innerText =
      "Twitter widget could not be loaded.";
  } else {
    count += 1;
  }
}, 1000);

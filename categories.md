---
layout: default
title: Categories
---

# Categories

Browse all posts by categories.

{% for category in site.categories %}
  <h3>{{ category[0] | upcase }}</h3>
  <ul>
    {% assign sorted_posts = category[1] | sort:"post-order" %}
    {% for post in sorted_posts %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}
---
title: '[Layout] Two columns'

---

---
title: '[Layout] Two columns'
---

::::info
**Two column effect cannot be previewed.**
**Apply the template to display the two-column effect.**

:::spoiler Click to learn more :wave:
Set column properties and quantities:

```
  column-count: 2;
  column-gap: 20px;
  max-width: 100%;
```

Set to cancel the text layout width restriction. Removing the syntax will restore HackMD's default text layout width:

```
.markdown-body, .ui-infobar {
    max-width: unset !important;
}
```

To ensure that Markdown syntax works correctly within a `div` container, please add a blank line before starting to write Markdown.

:pencil2: Feel free to customize it.
:::
::::

<style>
.two-column-layout {
  column-count: 2; /* Set column number */
  column-gap: 20px;
  max-width: 100%;
  overflow: hidden;
}

/* Media query for mobile devices */
@media (max-width: 768px) {
  .two-column-layout {
    column-count: 1; /* Switch to single column on small screens */
    column-gap: 0;   /* Optional: Set gap to 0 for single column */
  }
}

.markdown-body, .ui-infobar {
    max-width: unset !important;
}

.two-column-layout ul, 
.two-column-layout ol {
  margin: 0;
  padding-left: 20px;
}

.two-column-layout strong {
  font-weight: bold;
}

.two-column-layout em {
  font-style: italic;
}
    
.two-column-layout h1,
.two-column-layout h2,
.two-column-layout h3,
.two-column-layout h4,
.two-column-layout h5,
.two-column-layout h6 {
    margin-top: 0;    
}
</style>

# The Essential Benefits of Weekly Team Meetings 🙋

In the fast-paced world of project management, effective communication and collaboration are critical to achieving success. One of the most effective strategies for ensuring that a project stays on track is for teams to meet at least once a week. Regular weekly meetings offer numerous benefits that can significantly enhance the likelihood of a project's success.

<div class="two-column-layout">

![image](https://hackmd.io/_uploads/SyHssWcKR.png =600x)

### Here’s why maintaining a weekly meeting schedule is essential:

- Facilitates Problem-Solving
- Strengthens Team Cohesion
- Monitors Progress and Sets Goals
- Improves Communication
- Encourages Accountability
</div>

&nbsp;

## Check it out :+1:

<div class="two-column-layout">

### Facilitates Problem-Solving

Meetings provide a platform for discussing challenges and finding effective solutions collaboratively.

### Strengthens Team Cohesion

Regular meetings build strong relationships and enhance collaboration, leading to a more unified team.

</div>

<div class="two-column-layout">

### Monitors Progress and Sets Goals 🎯

Weekly check-ins help track progress, address issues, and set short-term goals, keeping the project on track.

### Improves Communication 👍

Structured meetings facilitate clear updates and expectation management, reducing misunderstandings.

</div>

<div class="two-column-layout">

### Encourages Accountability

Regular reporting fosters responsibility and helps team members stay focused on meeting deadlines.

### But if your team can't get together...

**That's why we needs HackMD!**
![](https://i.imgur.com/czzaRnH.png =500x)

</div>

// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "post-tutorial-a-diffusion-model-for-image-restoration",
      
        title: "Tutorial - A Diffusion Model for Image Restoration",
      
      description: "Theoretical Foundations of SDE-Based Diffusion Models for Image Restoration",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2024/irsde/";
        
      },
    },{id: "news-i-have-initiated-a-new-project-genotes-where-i-am-publising-some-of-my-notes-on-genai-models-as-a-blog-series",
          title: 'I have initiated a new project, Genotes, where I am publising some of...',
          description: "",
          section: "News",},{id: "news-i-will-be-co-chairing-the-agu-oral-amp-amp-poster-session-harnessing-geospatial-ai-geoai-best-practices-validation-and-real-world-applications-join-us-to-explore-the-advancements-in-ai-foundation-models-for-earth-sciences",
          title: 'I will be co-chairing the AGU oral &amp;amp;amp; poster session: “Harnessing Geospatial AI...',
          description: "",
          section: "News",},{id: "news-i-am-hiring-a-senior-research-scientist-full-time-and-a-phd-research-intern-6-months-ping-an-silicon-valley-lab-both-roles-will-focus-on-developing-multimodal-vision-language-models-vlms-aimed-at-addressing-climate-change-challenges-email-me-if-you-are-interested",
          title: 'I am hiring a (Senior) Research Scientist (full-time) and a PhD Research Intern...',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];

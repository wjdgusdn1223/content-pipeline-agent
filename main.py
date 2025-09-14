from crewai.flow.flow import Flow, listen, start, router, and_, or_
from crewai import Agent, LLM
from pydantic import BaseModel, Field
from tools import web_search_tool
from models import BlogPost, Tweet, LinkedInPost, Score
from seo_crew import SeoCrew
from virality_crew import ViralityCrew


class ContentPipelineState(BaseModel):

    # Inputs
    content_type: str = ""
    topic: str = ""

    # Internal
    max_length: int = 0
    score: int = 0
    research: str = ""
    score: Score | None = None
    try_count: int = 1
    try_limit: int = 5

    # Content
    blog_post: BlogPost | None = None
    tweet: Tweet | None = None
    linkedin_post: LinkedInPost | None = None


class ContentPipelineFlow(Flow[ContentPipelineState]):

    @start()
    def init_content_pipeline(self):
        if self.state.content_type not in ["tweet", "blog", "linkedin"]:
            raise ValueError("The content type is wrong")

        if self.state.topic == "":
            raise ValueError("The topic can't be blank")

        if self.state.content_type == "tweet":
            self.state.max_length = 150
        elif self.state.content_type == "blog":
            self.state.max_length = 2000
        elif self.state.content_type == "linkedin":
            self.state.max_length = 1000

    @listen(init_content_pipeline)
    def conduct_research(self):
        researcher = Agent(
            role="Head Researcher",
            backstory="You're like a digital detective who loves digging up fascinating facts and insights. You have a knack for finding the good stuff that others miss.",
            goal=f"Find the most interesting and useful info about {self.state.topic}",
            tools=[web_search_tool],
            verbose=True,
        )

        self.state.research = researcher.kickoff(
            f"Find the most interesting and useful info about {self.state.topic}"
        )

        return True

    @router(conduct_research)
    def conduct_research_router(self):
        content_type = self.state.content_type

        if content_type == "blog":
            return "make_blog"
        elif content_type == "tweet":
            return "make_tweet"
        else:
            return "make_linkedin_post"

    @listen(or_("make_blog", "remake_blog"))
    def handle_make_blog(self):
        # if blog post has been made, show the old one to the ai and ask it to improve, else
        # just ask to create.
        blog_post = self.state.blog_post

        llm = LLM(model="openai/o4-mini", response_format=BlogPost)

        if blog_post == None:
            result = llm.call(
                f"""
                     Make a blog post with SEO practices on the topic {self.state.topic} using the following research:

                     <research>
                     ================
                     {self.state.research}
                     ================
                     </research>

                     The maximum length of the content is {self.state.max_length}.
                     Post content should never exceed the maximum length.
            """
            )
        else:
            result = llm.call(
                f"""
                     You wrote this blog post on {self.state.topic}, but it does not have a good SEO score because of {self.state.score.reason}.

                     Improve it.
                     
                     <blog post>
                     ================
                     {self.state.blog_post.model_dump_json()}
                     ================
                     </blog post>
                     
                     Use the following research.

                     <research>
                     ================
                     {self.state.research}
                     ================
                     </research>

                     The maximum length of the content is {self.state.max_length}.
                     Post content should never exceed the maximum length.
            """
            )

        self.state.blog_post = BlogPost.model_validate_json(result)
        print(self.state.blog_post)

    @listen(or_("make_tweet", "remake_tweet"))
    def handle_make_tweet(self):
        # if tweet has been made, show the old one to the ai and ask it to improve, else
        # just ask to create.
        tweet = self.state.tweet

        llm = LLM(model="openai/o4-mini", response_format=Tweet)

        if tweet == None:
            result = llm.call(
                f"""
                     Make a tweet that can go viral on the topic {self.state.topic} using the following research:

                     <research>
                     ================
                     {self.state.research}
                     ================
                     </research>

                     The maximum length of the content is {self.state.max_length}.
            """
            )
        else:
            result = llm.call(
                f"""
                     You wrote this tweet on {self.state.topic}, but it does not have a good virality score because of {self.state.score.reason}.

                     Improve it.
                     
                     <tweet>
                     ================
                     {self.state.tweet.model_dump_json()}
                     ================
                     </tweet>
                     
                     Use the following research.

                     <research>
                     ================
                     {self.state.research}
                     ================
                     </research>

                     The maximum length of the content is {self.state.max_length}.
            """
            )

        self.state.tweet = Tweet.model_validate_json(result)
        print(self.state.tweet)

    @listen(or_("make_linkedin_post", "remake_linkedin_post"))
    def handle_make_linkedin_post(self):
        # if linkedin post has been made, show the old one to the ai and ask it to improve, else
        # just ask to create.
        linkedin_post = self.state.linkedin_post

        llm = LLM(model="openai/o4-mini", response_format=LinkedInPost)

        if linkedin_post == None:
            result = llm.call(
                f"""
                     Make a linkedin post that can go viral on the topic {self.state.topic} using the following research:

                     <research>
                     ================
                     {self.state.research}
                     ================
                     </research>

                     The maximum length of the content is {self.state.max_length}.
            """
            )
        else:
            result = llm.call(
                f"""
                     You wrote this linkedin post on {self.state.topic}, but it does not have a good virality score because of {self.state.score.reason}.

                     Improve it.
                     
                     <linkedin post>
                     ================
                     {self.state.linkedin_post.model_dump_json()}
                     ================
                     </linkedin post>
                     
                     Use the following research.

                     <research>
                     ================
                     {self.state.research}
                     ================
                     </research>

                     The maximum length of the content is {self.state.max_length}.
            """
            )

        self.state.linkedin_post = LinkedInPost.model_validate_json(result)
        print(self.state.linkedin_post)

    @listen(handle_make_blog)
    def check_seo(self):
        print(f"length: {len("".join(self.state.blog_post.sections))}")
        if (
            len(("".join(self.state.blog_post.sections)).replace(" ", ""))
            > self.state.max_length
        ):
            self.state.score = Score()
            self.state.score.reason = (
                f"The content is too long. max length is {self.state.max_length}."
            )
        else:
            result = (
                SeoCrew()
                .crew()
                .kickoff(
                    inputs={
                        "topic": self.state.topic,
                        "blog_post": self.state.blog_post.model_dump_json(),
                    }
                )
            )
            self.state.score = result.pydantic
        print(self.state.score)

    @listen(or_(handle_make_tweet, handle_make_linkedin_post))
    def check_virality(self):
        content = None

        if self.state.content_type == "tweet":
            content = self.state.tweet.content
        else:
            content = self.state.linkedin_post.content

        if len(content.replace(" ", "")) > self.state.max_length:
            self.state.score = Score()
            self.state.score.reason = (
                f"The content is too long. max length is {self.state.max_length}."
            )
        else:
            result = (
                ViralityCrew()
                .crew()
                .kickoff(
                    inputs={
                        "topic": self.state.topic,
                        "content_type": self.state.content_type,
                        "content": content,
                    }
                )
            )
            self.state.score = result.pydantic
        print(self.state.score)
        print(f"try count: {self.state.try_count}")

    @router(or_(check_seo, check_virality))
    def score_router(self):
        self.state.try_count = self.state.try_count + 1
        content_type = self.state.content_type
        score = self.state.score

        if score.score >= 8 or self.state.try_count == self.state.try_limit:
            return "check_passed"
        else:
            if content_type == "blog":
                return "remake_blog"
            elif content_type == "linkedin":
                return "remake_linkedin_post"
            else:
                return "remake_tweet"

    @listen("check_passed")
    def finalize_content(self):
        print("Finalizing content")


flow = ContentPipelineFlow()

# flow.plot()
flow.kickoff(
    inputs={
        "content_type": "tweet",
        "topic": "korea hyundai workers",
    }
)

/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the BSD license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const MarkdownBlock = CompLibrary.MarkdownBlock;
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

const bash = (...args) => `~~~bash\n${String.raw(...args)}\n~~~`;

class HomeSplash extends React.Component {
  render() {
    const {siteConfig, language = ''} = this.props;
    const {baseUrl, docsUrl} = siteConfig;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

    const Announcement = () => (
      <div className="announcement">
        <p>We are excited to announce the release of Opacus 1.0.</p>
        <p>This release packs in lot of new features and bug fixes, and most importantly, also brings forth new APIs that are simpler, more modular, and easily extensible.</p>
        <p>See our <a href="https://github.com/pytorch/opacus/releases/tag/v1.0.0">Release Notes</a> for more details. <a href="https://github.com/pytorch/opacus/blob/main/Migration_Guide.md" target="_blank">Migration Guide</a></p>
      </div>
    );

    const SplashContainer = props => (
      <div className="homeContainer">
        <Announcement />
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const Logo = props => (
      <div className="splashLogo">
        <img
          src={props.img_src}
          alt="Project Logo"
          className="primaryLogoImage"
        />
      </div>
    );

    const ProjectTitle = () => (
      <h2 className="projectTitle">
        <small>{siteConfig.tagline}</small>
      </h2>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
        </div>
      </div>
    );

    const Button = props => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
        <Logo img_src={siteConfig.logo} />
        <div className="inner">
          <ProjectTitle siteConfig={siteConfig} />
          <PromoSection>
            <Button href={docUrl('introduction.html')}>Introduction</Button>
            <Button href={'#quickstart'}>Get Started</Button>
            <Button href={`${baseUrl}tutorials/`}>Tutorials</Button>
          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}

class Index extends React.Component {
  render() {
    const {config: siteConfig, language = ''} = this.props;
    const {baseUrl} = siteConfig;

    const Block = props => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}>
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );
    // getStartedSection
    const pre = '```';
    // Example for model fitting
    const createModelExample = `${pre}python
# define your components as usual
model = Net()
optimizer = SGD(model.parameters(), lr=0.05)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024)

# enter PrivacyEngine
privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)
# Now it's business as usual
    `;

    //
    const QuickStart = () => (
      <div
        className="productShowcaseSection"
        id="quickstart"
        style={{textAlign: 'center'}}>
        <h2>Get Started</h2>
        <Container>
          <ol>
            <li>
              <h4>Installing Opacus:</h4>
              <a>Via pip:</a>
              <MarkdownBlock>{bash`pip install opacus`}</MarkdownBlock>
              <a>From source:</a>
              <MarkdownBlock>{bash`
git clone https://github.com/pytorch/opacus.git
cd opacus
pip install -e .
                `}</MarkdownBlock>
            </li>
            <li>
              <h4>Getting started</h4>
              <a>
                Training with differential privacy is as simple as instantiating
                a PrivacyEngine:
              </a>
              <MarkdownBlock>{createModelExample}</MarkdownBlock>
            </li>
          </ol>
        </Container>
      </div>
    );

    const Features = () => (
      <div className="productShowcaseSection" style={{textAlign: 'center'}}>
        <h2>Key Features</h2>
        <Block layout="threeColumn">
          {[
            {
              content:
                'Vectorized per-sample gradient computation that is 10x faster than microbatching',
              image: `${baseUrl}img/expanding_arrows.svg`,
              imageAlign: 'top',
              imageAlt: 'Scalable logo',
              title: 'Scalable',
            },
            {
              content:
                'Supports most types of PyTorch models and can be used with minimal modification to the original neural network.',
              image: `${baseUrl}img/pytorch_logo.svg`,
              imageAlign: 'top',
              imageAlt: 'PyTorch logo',
              title: 'Built on PyTorch',
            },
            {
              content:
                'Open source, modular API for differential privacy research. Everyone is welcome to contribute.',
              image: `${baseUrl}img/modular.svg`,
              imageAlign: 'top',
              imageAlt: 'Extensible logo',
              title: 'Extensible',
            },
          ]}
        </Block>
      </div>
    );

    const Showcase = () => {
      if ((siteConfig.users || []).length === 0) {
        return null;
      }

      const showcase = siteConfig.users
        .filter(user => user.pinned)
        .map(user => (
          <a href={user.infoLink} key={user.infoLink}>
            <img src={user.image} alt={user.caption} title={user.caption} />
          </a>
        ));

      const pageUrl = page => baseUrl + (language ? `${language}/` : '') + page;

      return (
        <div className="productShowcaseSection paddingBottom">
          <h2>Who is Using This?</h2>
          <p>This project is used by all these people</p>
          <div className="logos">{showcase}</div>
          <div className="more-users">
            <a className="button" href={pageUrl('users.html')}>
              More {siteConfig.title} Users
            </a>
          </div>
        </div>
      );
    };

    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="landingPage mainContainer">
          <Features />
          <QuickStart />
        </div>
      </div>
    );
  }
}

module.exports = Index;

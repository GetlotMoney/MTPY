"""
Claude Opus 4.7 - CUB 200 鸟类完整描述
=========================================
新格式: 7 个字符串/类
  [0] beak    - 喙: 形状、长度、颜色
  [1] head    - 头: 颜色、冠羽、眼环、面纹
  [2] body    - 身: 上下体颜色、纹路
  [3] wings   - 翅: 颜色、翅斑、形状
  [4] tail    - 尾: 长度、形状、颜色
  [5] legs    - 腿: 颜色、长度
  [6] caption - 整体一句话 "A photo of a {class}, a ..." (CLIP 风格)

使用:
  python tools/generate_claude_descriptions.py
  → 生成 data/gpt4_data/cub_claude.pt

替换原 GPT-4 数据:
  把 cub_claude.pt 重命名为 cub.pt 即可直接用
  (训练脚本默认读 cub.pt, schema 完全兼容)
"""

CUB_CLAUDE = {}

# ============== A ==============
CUB_CLAUDE["acadian_flycatcher"] = [
    "Acadian flycatcher is a bird with beak that is medium length, broad and flat at the base, dark upper mandible and pale orange-pink lower mandible",
    "Acadian flycatcher is a bird with head that is olive-green crown blending into the back, distinct white eye-ring, no crest",
    "Acadian flycatcher is a bird with body that is olive-green upperparts, pale yellowish-white underparts with a faint olive wash across the breast",
    "Acadian flycatcher is a bird with wings that are dusky with two clean buffy-white wing bars and pale edges on the tertials",
    "Acadian flycatcher is a bird with tail that is medium length, square-tipped, olive-brown with paler edges",
    "Acadian flycatcher is a bird with legs that are short, dark gray to blackish, suited for perching upright",
    "A photo of an acadian flycatcher, a small olive-and-white empidonax flycatcher with bold white eye-ring and two pale wing bars",
]
CUB_CLAUDE["american_crow"] = [
    "American crow is a bird with beak that is large, thick, slightly curved, entirely glossy black with bristly nasal feathers",
    "American crow is a bird with head that is uniformly glossy black, smooth without crest, dark eye blending into face",
    "American crow is a bird with body that is entirely solid glossy black with subtle purple-blue iridescence in good light",
    "American crow is a bird with wings that are broad, rounded, all glossy black with fingered primaries spreading in flight",
    "American crow is a bird with tail that is medium length, fan-shaped when spread, square-tipped, all black",
    "American crow is a bird with legs that are sturdy, entirely black, suited for walking on the ground",
    "A photo of an american crow, a medium-sized all-black corvid with broad wings and a fan-shaped tail",
]
CUB_CLAUDE["american_goldfinch"] = [
    "American goldfinch is a bird with beak that is short, conical, pinkish-orange in breeding males and dull horn-colored in winter",
    "American goldfinch is a bird with head that is bright lemon-yellow in breeding males with a jet-black forehead patch, duller olive-yellow in females",
    "American goldfinch is a bird with body that is brilliant canary-yellow on breast and back in breeding males, olive-brown in females and winter birds",
    "American goldfinch is a bird with wings that are jet-black with bold white wing bars and white edges on flight feathers",
    "American goldfinch is a bird with tail that is short, notched, black with white outer edges visible in flight",
    "American goldfinch is a bird with legs that are short, dull pinkish-buff, slender",
    "A photo of an american goldfinch, a tiny bright yellow finch with black wings, white wing bars, and a black forehead in males",
]
CUB_CLAUDE["american_pipit"] = [
    "American pipit is a bird with beak that is slender, pointed, dark gray with a slightly paler base on the lower mandible",
    "American pipit is a bird with head that is plain grayish-brown crown, faint pale eyebrow, dark eye-line, no eye-ring",
    "American pipit is a bird with body that is grayish-brown above, pale buffy-white below with fine dark streaking on the breast and flanks",
    "American pipit is a bird with wings that are brown with two thin pale wing bars, fairly pointed in shape",
    "American pipit is a bird with tail that is medium length, dark brown with conspicuous white outer tail feathers",
    "American pipit is a bird with legs that are long, slender, dark blackish, often walking rather than hopping",
    "A photo of an american pipit, a slim ground-dwelling brownish songbird with streaked breast, white outer tail, and constantly bobbing tail",
]
CUB_CLAUDE["american_redstart"] = [
    "American redstart is a bird with beak that is small, slender, pointed, entirely black",
    "American redstart is a bird with head that is glossy black in adult males, gray with a pale eye-ring in females and immatures",
    "American redstart is a bird with body that is jet-black upperparts and breast in males with white belly, females olive-gray above and pale below",
    "American redstart is a bird with wings that are black with bright orange flash patches at the base of flight feathers in males, yellow patches in females",
    "American redstart is a bird with tail that is medium, often fanned, black with bright orange basal patches in males, yellow patches in females",
    "American redstart is a bird with legs that are dark blackish-gray, slender, fairly short",
    "A photo of an american redstart, a small black warbler with orange flash patches on wings and tail, females replace orange with yellow",
]
CUB_CLAUDE["american_three_toed_woodpecker"] = [
    "American three toed woodpecker is a bird with beak that is medium length, straight, chisel-tipped, blackish-gray",
    "American three toed woodpecker is a bird with head that is black with white eyebrow and white moustache stripes, males show a yellow crown patch",
    "American three toed woodpecker is a bird with body that is mostly black above with a barred white-and-black back panel, white below with barred flanks",
    "American three toed woodpecker is a bird with wings that are black with sparse white spots on flight feathers and a white-barred back",
    "American three toed woodpecker is a bird with tail that is stiff, black with white outer feathers, used as a brace against tree trunks",
    "American three toed woodpecker is a bird with legs that are short, gray, with three toes (two forward, one back) instead of four",
    "A photo of an american three toed woodpecker, a small black-and-white woodpecker with barred back panel and a yellow crown in males",
]
CUB_CLAUDE["anna_hummingbird"] = [
    "Anna hummingbird is a bird with beak that is medium length for a hummingbird, straight, slender, entirely black",
    "Anna hummingbird is a bird with head that is metallic emerald-green in females, with adult males showing a brilliant rose-pink iridescent crown and gorget",
    "Anna hummingbird is a bird with body that is bright iridescent green upperparts, grayish-white underparts with green flecking on the sides",
    "Anna hummingbird is a bird with wings that are dark, narrow, pointed, beating extremely fast and creating a humming sound",
    "Anna hummingbird is a bird with tail that is dark gray-green, slightly forked in males, square in females, often pumped while hovering",
    "Anna hummingbird is a bird with legs that are tiny, dark, used only for perching not walking",
    "A photo of an anna hummingbird, a small iridescent green hummer where adult males display a flashing rose-magenta head and throat",
]
CUB_CLAUDE["artic_tern"] = [
    "Artic tern is a bird with beak that is slender, pointed, blood-red in breeding adults, blackish in winter and immatures",
    "Artic tern is a bird with head that is white with a sharply defined solid black cap covering the crown and nape in breeding plumage",
    "Artic tern is a bird with body that is pale gray above and below, very slightly grayer underparts than other terns, white rump",
    "Artic tern is a bird with wings that are long, narrow, pale gray with translucent primaries showing as a pale flash from below",
    "Artic tern is a bird with tail that is deeply forked with very long streamers, white with gray edges, characteristic 'sea swallow' shape",
    "Artic tern is a bird with legs that are very short, blood-red in breeding plumage, almost invisible in flight",
    "A photo of an arctic tern, a slender pale-gray tern with black cap, blood-red bill, and very long forked tail streamers",
]
CUB_CLAUDE["baird_sparrow"] = [
    "Baird sparrow is a bird with beak that is short, conical, sharply pointed, pale pinkish-horn with a darker upper mandible",
    "Baird sparrow is a bird with head that is buffy-yellow with a distinctive ochre-yellow stripe down the center of the crown, fine dark streaks on cheeks",
    "Baird sparrow is a bird with body that is buffy-brown above with crisp dark streaks, pale buff below with a thin necklace of fine streaks across the upper breast",
    "Baird sparrow is a bird with wings that are brown with rufous edges on the wing coverts, no obvious wing bars",
    "Baird sparrow is a bird with tail that is short, slightly notched, brown with paler outer edges",
    "Baird sparrow is a bird with legs that are pinkish-buff, fairly short, suited for skulking in tall prairie grass",
    "A photo of a baird sparrow, a secretive prairie sparrow with an ochre-yellow head stripe and a fine breast necklace of streaks",
]
CUB_CLAUDE["baltimore_oriole"] = [
    "Baltimore oriole is a bird with beak that is long, sharply pointed, slightly downcurved, blue-gray with a black tip",
    "Baltimore oriole is a bird with head that is solid jet-black hood covering the entire head, throat and upper back in adult males",
    "Baltimore oriole is a bird with body that is brilliant flame-orange below and on the rump in males, females and immatures yellower-orange with grayish back",
    "Baltimore oriole is a bird with wings that are black with a single bold white wing bar and white edges on flight feathers",
    "Baltimore oriole is a bird with tail that is medium length, black with bright orange basal corners forming a 'T' pattern from below",
    "Baltimore oriole is a bird with legs that are blue-gray, fairly long, suited for hopping along branches",
    "A photo of a baltimore oriole, a striking flame-orange-and-black songbird where adult males have a fully black hood",
]
CUB_CLAUDE["bank_swallow"] = [
    "Bank swallow is a bird with beak that is very short, broad at the base, dark blackish, adapted for catching insects in flight",
    "Bank swallow is a bird with head that is plain brown crown blending into the back, white throat, no obvious eye markings",
    "Bank swallow is a bird with body that is dull brown upperparts, white underparts with a distinctive brown breast band across the chest",
    "Bank swallow is a bird with wings that are brown, long, narrow and pointed, built for agile aerial maneuvering",
    "Bank swallow is a bird with tail that is short, slightly notched, brown",
    "Bank swallow is a bird with legs that are very short, dark, mostly hidden in flight, weak for perching",
    "A photo of a bank swallow, a small brown-and-white swallow with a clean brown breast band across the white underparts",
]
CUB_CLAUDE["barn_swallow"] = [
    "Barn swallow is a bird with beak that is very short, broad-based, blackish, well-suited for aerial insect catching",
    "Barn swallow is a bird with head that is glossy steel-blue crown, with a distinctive deep rusty-cinnamon forehead and throat",
    "Barn swallow is a bird with body that is glossy steel-blue upperparts, warm rusty-buff underparts with paler belly",
    "Barn swallow is a bird with wings that are long, very pointed, glossy blue-black, built for sustained agile flight",
    "Barn swallow is a bird with tail that is deeply forked with long outer streamers, blue-black with white spots near the base",
    "Barn swallow is a bird with legs that are very short, dark, weak, used only for perching on wires or ledges",
    "A photo of a barn swallow, a sleek blue-and-rusty swallow with a deeply forked tail and long outer streamers",
]
CUB_CLAUDE["bay_breasted_warbler"] = [
    "Bay breasted warbler is a bird with beak that is small, slender, sharply pointed, dark gray-black",
    "Bay breasted warbler is a bird with head that is rich chestnut crown in breeding males with a black face mask, pale creamy patch on the side of the neck",
    "Bay breasted warbler is a bird with body that is grayish-olive above with black streaks, with deep chestnut-bay flanks and throat in breeding males, dull greenish in fall",
    "Bay breasted warbler is a bird with wings that are dark with two prominent white wing bars",
    "Bay breasted warbler is a bird with tail that is short, dark with white spots on the outer tail feathers",
    "Bay breasted warbler is a bird with legs that are dark, short, suited for foraging high in conifers",
    "A photo of a bay breasted warbler, a small warbler with chestnut crown and flanks, black face mask, and creamy neck patch in breeding plumage",
]
CUB_CLAUDE["belted_kingfisher"] = [
    "Belted kingfisher is a bird with beak that is large, long, dagger-like, entirely black, perfectly built for spearing fish",
    "Belted kingfisher is a bird with head that is large with a shaggy double-pointed blue-gray crest, white collar around the neck, white throat",
    "Belted kingfisher is a bird with body that is slate-blue-gray above with a blue chest band, white below, females additionally have a rusty-orange belly band",
    "Belted kingfisher is a bird with wings that are blue-gray, rounded, with white spots and a white patch visible in flight",
    "Belted kingfisher is a bird with tail that is medium length, blue-gray with white bars, often pumped while perched",
    "Belted kingfisher is a bird with legs that are very short, dark, with small feet adapted for perching not walking",
    "A photo of a belted kingfisher, a stocky blue-gray bird with a shaggy crest, dagger bill, and white collar; females additionally show a rusty belly band",
]
CUB_CLAUDE["bewick_wren"] = [
    "Bewick wren is a bird with beak that is long, slender, slightly downcurved, dark gray-brown",
    "Bewick wren is a bird with head that is plain brown crown with a long, conspicuous white eyebrow stripe contrasting against a darker eye-line",
    "Bewick wren is a bird with body that is plain warm brown above, clean grayish-white below without streaks or barring",
    "Bewick wren is a bird with wings that are brown with very faint dark barring, short and rounded",
    "Bewick wren is a bird with tail that is long, often cocked vertically, brown with dark barring and white spots on the outer corners",
    "Bewick wren is a bird with legs that are pinkish-gray, fairly long, suited for hopping through brush",
    "A photo of a bewick wren, a slim brown-and-white wren with a long bold white eyebrow and a long barred tail often held cocked upward",
]
CUB_CLAUDE["black_and_white_warbler"] = [
    "Black and white warbler is a bird with beak that is slender, slightly downcurved, sharply pointed, all black",
    "Black and white warbler is a bird with head that is boldly striped black-and-white with a white central crown stripe and white eyebrow",
    "Black and white warbler is a bird with body that is entirely streaked black-and-white above and below, looking 'zebra-striped' overall",
    "Black and white warbler is a bird with wings that are black with two bold white wing bars",
    "Black and white warbler is a bird with tail that is medium length, black with white spots on the outer feathers",
    "Black and white warbler is a bird with legs that are dark, short, with strong feet adapted to climb tree trunks like a nuthatch",
    "A photo of a black and white warbler, a small zebra-striped warbler that creeps along trunks and limbs like a nuthatch",
]
CUB_CLAUDE["black_billed_cuckoo"] = [
    "Black billed cuckoo is a bird with beak that is long, slightly downcurved, entirely black",
    "Black billed cuckoo is a bird with head that is plain warm brown with a narrow red eye-ring around a dark eye, no contrasting face pattern",
    "Black billed cuckoo is a bird with body that is plain warm brown above, entirely white below without obvious markings",
    "Black billed cuckoo is a bird with wings that are warm brown lacking the rusty primaries seen in yellow-billed cuckoos",
    "Black billed cuckoo is a bird with tail that is very long, brown above with small white tips on the outer feathers",
    "Black billed cuckoo is a bird with legs that are short, dark gray, with two toes forward and two back",
    "A photo of a black billed cuckoo, a slim long-tailed brown-and-white cuckoo with a red eye-ring and an all-black bill",
]
CUB_CLAUDE["black_capped_vireo"] = [
    "Black capped vireo is a bird with beak that is short, thick, slightly hooked, dark with a paler base",
    "Black capped vireo is a bird with head that is solid glossy black in males with bold white spectacles around the red-brown eye, gray-headed in females",
    "Black capped vireo is a bird with body that is olive-green above, white below with yellow-tinged flanks",
    "Black capped vireo is a bird with wings that are olive with two pale yellow wing bars",
    "Black capped vireo is a bird with tail that is short, olive-brown, with paler edges",
    "Black capped vireo is a bird with legs that are dark gray, sturdy, fairly short",
    "A photo of a black capped vireo, a tiny vireo with a glossy black hood and bold white spectacles in males, females show gray hood",
]
CUB_CLAUDE["black_footed_albatross"] = [
    "Black footed albatross is a bird with beak that is huge, long, hooked, dark gray to blackish, built for tearing food at sea",
    "Black footed albatross is a bird with head that is mostly sooty-brown with a small whitish ring around the base of the bill and pale forehead",
    "Black footed albatross is a bird with body that is uniformly sooty-brown above and below, often with white undertail coverts in older birds",
    "Black footed albatross is a bird with wings that are extremely long, narrow, pointed, sooty-brown, built for effortless dynamic soaring over open ocean",
    "Black footed albatross is a bird with tail that is short relative to wing length, sooty-brown, square-tipped",
    "Black footed albatross is a bird with legs that are short, blackish, with webbed feet for swimming",
    "A photo of a black footed albatross, a large sooty-brown seabird with extremely long narrow wings and a pale ring at the base of the heavy bill",
]
CUB_CLAUDE["black_tern"] = [
    "Black tern is a bird with beak that is slender, sharply pointed, all black",
    "Black tern is a bird with head that is solid black in breeding plumage extending down to the breast, white in winter with a dark cap and ear patch",
    "Black tern is a bird with body that is dark slate-gray above and entirely black below in breeding plumage, white below in non-breeding",
    "Black tern is a bird with wings that are slate-gray, long, narrow and pointed, built for buoyant aerial foraging",
    "Black tern is a bird with tail that is medium, slightly forked, gray",
    "Black tern is a bird with legs that are very short, dark reddish-black",
    "A photo of a black tern, a small dark tern that is solid black on head and underparts in breeding plumage with slate-gray wings",
]
CUB_CLAUDE["black_throated_blue_warbler"] = [
    "Black throated blue warbler is a bird with beak that is short, slender, sharply pointed, all black",
    "Black throated blue warbler is a bird with head that is deep slate-blue in males with a coal-black face and throat, females plain olive with a pale eyebrow",
    "Black throated blue warbler is a bird with body that is deep blue above and white below in males with black flanks; females plain olive above and buffy-yellow below",
    "Black throated blue warbler is a bird with wings that are blue in males with a small but distinctive white square at the base of primaries; same white patch in females",
    "Black throated blue warbler is a bird with tail that is short, blue-black in males, olive-brown in females",
    "Black throated blue warbler is a bird with legs that are dark, short, slender",
    "A photo of a black throated blue warbler, a small warbler where males show deep blue upperparts, black throat, and a white wing square; females are plain olive with the same white wing patch",
]
CUB_CLAUDE["black_throated_sparrow"] = [
    "Black throated sparrow is a bird with beak that is short, conical, sharp, dark gray with a paler base",
    "Black throated sparrow is a bird with head that is gray with two crisp white stripes (eyebrow and submoustache) bracketing a black ear patch",
    "Black throated sparrow is a bird with body that is plain gray above, white below with a contrasting solid black bib on the throat and upper breast",
    "Black throated sparrow is a bird with wings that are gray-brown, plain, without obvious wing bars",
    "Black throated sparrow is a bird with tail that is medium length, dark gray-black with white outer corners",
    "Black throated sparrow is a bird with legs that are pinkish-gray, short to medium length",
    "A photo of a black throated sparrow, a clean gray-and-white desert sparrow with bold white face stripes and a sharp black throat bib",
]
CUB_CLAUDE["blue_grosbeak"] = [
    "Blue grosbeak is a bird with beak that is very large, thick, conical, silver-gray with a darker upper mandible",
    "Blue grosbeak is a bird with head that is deep cobalt-blue all over in adult males, warm cinnamon-brown in females",
    "Blue grosbeak is a bird with body that is deep cobalt-blue overall in males, warm cinnamon-brown above and buffy below in females",
    "Blue grosbeak is a bird with wings that are blue-black in males with two distinctive rusty-cinnamon wing bars; same wing bar pattern in browner females",
    "Blue grosbeak is a bird with tail that is medium, dark blue or brown, often flicked or spread",
    "Blue grosbeak is a bird with legs that are dark gray-black, sturdy",
    "A photo of a blue grosbeak, a stocky dark-blue grosbeak with rusty wing bars and a massive silvery bill; females are warm cinnamon-brown",
]
CUB_CLAUDE["blue_headed_vireo"] = [
    "Blue headed vireo is a bird with beak that is short, thick, slightly hooked at the tip, dark gray",
    "Blue headed vireo is a bird with head that is blue-gray crown contrasting with bold white spectacles around the eye and a clean white throat",
    "Blue headed vireo is a bird with body that is olive-green back contrasting with the blue-gray head, white below with greenish-yellow flanks",
    "Blue headed vireo is a bird with wings that are dark olive with two crisp white wing bars",
    "Blue headed vireo is a bird with tail that is medium, dark olive, with paler edges",
    "Blue headed vireo is a bird with legs that are blue-gray, sturdy, medium length",
    "A photo of a blue headed vireo, a small vireo with a blue-gray hood, bold white spectacles, white throat, and yellow-tinged flanks",
]
CUB_CLAUDE["blue_jay"] = [
    "Blue jay is a bird with beak that is medium length, thick, sharply pointed, all black",
    "Blue jay is a bird with head that is bright sky-blue with a tall pointed crest and a black necklace extending up around the white face",
    "Blue jay is a bird with body that is bright blue above, grayish-white below, with a black necklace across the chest",
    "Blue jay is a bird with wings that are vivid blue with bold black barring and white wing patches and tips",
    "Blue jay is a bird with tail that is long, bright blue with bold black bars and white tips on the outer feathers",
    "Blue jay is a bird with legs that are black, fairly long and strong, suited for hopping on the ground or branches",
    "A photo of a blue jay, a striking crested blue corvid with white face, black necklace, and bold black-and-white wing and tail markings",
]
CUB_CLAUDE["blue_winged_warbler"] = [
    "Blue winged warbler is a bird with beak that is small, slender, sharply pointed, dark gray",
    "Blue winged warbler is a bird with head that is bright lemon-yellow with a thin black eye-line cutting across the face",
    "Blue winged warbler is a bird with body that is olive-yellow upperparts, bright yellow underparts including throat and belly",
    "Blue winged warbler is a bird with wings that are blue-gray with two clean white wing bars",
    "Blue winged warbler is a bird with tail that is short, blue-gray with white spots on the outer feathers",
    "Blue winged warbler is a bird with legs that are dark, slender, fairly short",
    "A photo of a blue winged warbler, a small bright-yellow warbler with a thin black eye-line and blue-gray wings with two white wing bars",
]
CUB_CLAUDE["boat_tailed_grackle"] = [
    "Boat tailed grackle is a bird with beak that is long, sharply pointed, slightly downcurved, all black",
    "Boat tailed grackle is a bird with head that is glossy iridescent blue-purple in males, plain warm brown in females, with pale eye in some populations",
    "Boat tailed grackle is a bird with body that is glossy iridescent blue-black overall in males, warm tawny-brown above and buff below in females",
    "Boat tailed grackle is a bird with wings that are glossy black in males, brown in females, fairly long",
    "Boat tailed grackle is a bird with tail that is very long, deeply keeled (V-shaped from behind, like a boat hull), iridescent black in males",
    "Boat tailed grackle is a bird with legs that are long, sturdy, all black",
    "A photo of a boat tailed grackle, a large coastal grackle where males show glossy iridescent blue-black plumage and a long V-shaped keel-tail; females are warm brown",
]
CUB_CLAUDE["bobolink"] = [
    "Bobolink is a bird with beak that is short, conical, sharply pointed, dark in breeding males and pinkish in females",
    "Bobolink is a bird with head that is mostly black in breeding males with a striking pale yellowish-buff nape patch, brown-and-buff striped in females",
    "Bobolink is a bird with body that is unique reverse pattern in breeding males—black below with white shoulders and rump, females are buffy with dark streaks",
    "Bobolink is a bird with wings that are mostly black with bright white shoulders and scapulars in males, brown with buff edges in females",
    "Bobolink is a bird with tail that is short, with stiff pointed tail feathers, black in males and brown in females",
    "Bobolink is a bird with legs that are pinkish-buff, fairly long for a sparrow-sized bird, walking on grass",
    "A photo of a bobolink, an unusual blackbird where breeding males are black below with bright white back and yellow nape, looking 'inverted'",
]
CUB_CLAUDE["bohemian_waxwing"] = [
    "Bohemian waxwing is a bird with beak that is short, slightly hooked, dark gray-black",
    "Bohemian waxwing is a bird with head that is warm cinnamon-brown with a tall pointed crest, black mask through the eye, and a black throat",
    "Bohemian waxwing is a bird with body that is soft gray-brown above and below, distinguishing chestnut undertail coverts",
    "Bohemian waxwing is a bird with wings that are dark gray with sharp white-and-yellow markings on the secondaries plus tiny red waxy tips on inner flight feathers",
    "Bohemian waxwing is a bird with tail that is medium, gray with a broad bright yellow tip",
    "Bohemian waxwing is a bird with legs that are short, dark gray, sturdy",
    "A photo of a bohemian waxwing, a sleek gray-brown waxwing with crest, black mask, chestnut undertail, white-and-yellow wing markings, and red waxy wing tips",
]
CUB_CLAUDE["brandt_cormorant"] = [
    "Brandt cormorant is a bird with beak that is long, hooked at the tip, dark gray with a pale yellowish base",
    "Brandt cormorant is a bird with head that is glossy dark green-black in adults, with a striking bright cobalt-blue throat patch in breeding plumage",
    "Brandt cormorant is a bird with body that is uniformly glossy dark blue-black overall, less iridescent in immatures",
    "Brandt cormorant is a bird with wings that are dark, broad, used for both flying and underwater swimming",
    "Brandt cormorant is a bird with tail that is medium length, stiff, dark blackish",
    "Brandt cormorant is a bird with legs that are dark, with fully webbed feet for diving",
    "A photo of a brandt cormorant, a glossy dark cormorant with a striking cobalt-blue throat patch in breeding adults",
]
CUB_CLAUDE["brewer_blackbird"] = [
    "Brewer blackbird is a bird with beak that is medium length, slender, sharply pointed, all black",
    "Brewer blackbird is a bird with head that is glossy purple-iridescent in males with a striking pale yellow eye, plain dull brown in females with dark eye",
    "Brewer blackbird is a bird with body that is glossy iridescent black with greenish reflections in males, plain medium-brown in females",
    "Brewer blackbird is a bird with wings that are glossy black in males, brown in females, fairly pointed",
    "Brewer blackbird is a bird with tail that is medium length, glossy black in males, brown in females",
    "Brewer blackbird is a bird with legs that are slender, all black, suited for walking on the ground",
    "A photo of a brewer blackbird, a slim glossy blackbird where males have iridescent black plumage and a piercing yellow eye, females are plain brown",
]
CUB_CLAUDE["brewer_sparrow"] = [
    "Brewer sparrow is a bird with beak that is small, conical, sharply pointed, pale pinkish-horn",
    "Brewer sparrow is a bird with head that is plainly streaked grayish-brown with a pale eye-ring, lacking strong face pattern unlike other Spizella sparrows",
    "Brewer sparrow is a bird with body that is sandy gray-brown above with fine dark streaks, plain grayish-buff below without streaks",
    "Brewer sparrow is a bird with wings that are brown with two faint pale wing bars",
    "Brewer sparrow is a bird with tail that is medium, slightly notched, brown",
    "Brewer sparrow is a bird with legs that are pinkish-buff, fairly short",
    "A photo of a brewer sparrow, a small drab sandy-gray sparrow with a finely streaked back, pale eye-ring, and overall plain face",
]
CUB_CLAUDE["bronzed_cowbird"] = [
    "Bronzed cowbird is a bird with beak that is thick, conical, sharply pointed, all black",
    "Bronzed cowbird is a bird with head that is glossy bronze-iridescent in males with a striking bright red eye, females glossy black with red eye too",
    "Bronzed cowbird is a bird with body that is glossy bronze-black with greenish wings in males, charcoal-gray in females",
    "Bronzed cowbird is a bird with wings that are dark with greenish iridescence, broad and fairly rounded",
    "Bronzed cowbird is a bird with tail that is medium, dark, often spread when displaying",
    "Bronzed cowbird is a bird with legs that are sturdy, all black",
    "A photo of a bronzed cowbird, a stocky glossy black-and-bronze blackbird with a piercing red eye and a 'bull-necked' silhouette",
]
CUB_CLAUDE["brown_creeper"] = [
    "Brown creeper is a bird with beak that is long, very slender, downcurved, dark with paler base",
    "Brown creeper is a bird with head that is streaked brown crown blending into the bark-like back, pale eyebrow over the eye",
    "Brown creeper is a bird with body that is camouflage-perfect mottled brown-and-buff streaked above, clean white below",
    "Brown creeper is a bird with wings that are brown with a buffy band across the flight feathers visible in flight",
    "Brown creeper is a bird with tail that is long, stiff, pointed, brown, used as a brace against tree trunks like a woodpecker",
    "Brown creeper is a bird with legs that are pinkish-buff, short with strong claws for climbing trunks",
    "A photo of a brown creeper, a tiny camouflaged brown-and-white tree-climber with a downcurved bill and a stiff pointed tail",
]
CUB_CLAUDE["brown_pelican"] = [
    "Brown pelican is a bird with beak that is enormous, very long, with an expandable throat pouch underneath, gray to yellowish",
    "Brown pelican is a bird with head that is white in adults with a yellow crown, pale rusty nape in breeding plumage, dark in juveniles",
    "Brown pelican is a bird with body that is dark gray-brown overall in adults, all-brown in juveniles, large and stocky",
    "Brown pelican is a bird with wings that are very long, broad, dark gray-brown, built for soaring just above the waves",
    "Brown pelican is a bird with tail that is short, dark, square-tipped",
    "Brown pelican is a bird with legs that are short, dark, with fully webbed feet for swimming",
    "A photo of a brown pelican, a huge dark seabird with an enormous bill and throat pouch, white head with yellow crown in adults",
]
CUB_CLAUDE["brown_thrasher"] = [
    "Brown thrasher is a bird with beak that is long, slender, slightly downcurved, dark gray with paler base",
    "Brown thrasher is a bird with head that is bright rufous-brown crown with a striking bright yellow eye and a pale gray cheek",
    "Brown thrasher is a bird with body that is bright rufous-brown above, white below with bold dark streaks running down the entire underparts",
    "Brown thrasher is a bird with wings that are rufous-brown with two narrow white wing bars",
    "Brown thrasher is a bird with tail that is very long, rufous-brown, often cocked or wagged",
    "Brown thrasher is a bird with legs that are pinkish-buff, fairly long, suited for hopping through dense brush",
    "A photo of a brown thrasher, a long-tailed rufous-brown mimid with a heavily streaked white breast and piercing yellow eye",
]


# ============== C ==============
CUB_CLAUDE["cactus_wren"] = [
    "Cactus wren is a bird with beak that is long, slender, slightly downcurved, dark gray-brown",
    "Cactus wren is a bird with head that is brown crown with a long bold white eyebrow stripe and a brown ear patch",
    "Cactus wren is a bird with body that is rich brown above with white spots, white below with a heavily black-spotted breast and barred flanks",
    "Cactus wren is a bird with wings that are brown with prominent black-and-white barring on flight feathers",
    "Cactus wren is a bird with tail that is long, brown with bold black-and-white barring, often fanned showing white tips",
    "Cactus wren is a bird with legs that are pinkish-gray, long and sturdy for hopping in cactus",
    "A photo of a cactus wren, a large brown wren with a bold white eyebrow, heavy black breast spotting, and a long barred tail",
]
CUB_CLAUDE["california_gull"] = [
    "California gull is a bird with beak that is medium length, yellow with both a black ring and a red spot near the tip",
    "California gull is a bird with head that is white in breeding adults, dusky-streaked in winter, with a dark eye and red orbital ring in breeding",
    "California gull is a bird with body that is white below, medium-dark gray mantle (between Herring and Western Gull), juveniles mottled brown",
    "California gull is a bird with wings that are medium gray with black wingtips having small white mirrors",
    "California gull is a bird with tail that is white in adults, dark-banded in immatures, square-tipped",
    "California gull is a bird with legs that are greenish-yellow to yellow, distinguishing from pink-legged gulls",
    "A photo of a california gull, a medium-gray gull with greenish-yellow legs, dark eye, and a yellow bill marked with both red and black spots",
]
CUB_CLAUDE["canada_warbler"] = [
    "Canada warbler is a bird with beak that is small, slender, sharply pointed, dark with paler lower mandible",
    "Canada warbler is a bird with head that is gray crown with bright yellow spectacles around the eyes",
    "Canada warbler is a bird with body that is plain gray above, brilliant lemon-yellow below with a striking black 'necklace' of streaks across the upper breast",
    "Canada warbler is a bird with wings that are plain gray without obvious wing bars",
    "Canada warbler is a bird with tail that is medium, gray, with white undertail coverts",
    "Canada warbler is a bird with legs that are pinkish-flesh, fairly long",
    "A photo of a canada warbler, a small bright-yellow warbler with gray upperparts, yellow spectacles, and a bold black necklace of streaks across the chest",
]
CUB_CLAUDE["cape_glossy_starling"] = [
    "Cape glossy starling is a bird with beak that is medium, sharp, all black",
    "Cape glossy starling is a bird with head that is glossy iridescent metallic blue-green with a piercing bright orange-yellow eye",
    "Cape glossy starling is a bird with body that is uniformly glossy iridescent metallic blue-green above and below, vivid in good light",
    "Cape glossy starling is a bird with wings that are glossy blue-green with darker bars on flight feathers",
    "Cape glossy starling is a bird with tail that is medium length, glossy blue-green, square-tipped",
    "Cape glossy starling is a bird with legs that are dark gray-black, sturdy",
    "A photo of a cape glossy starling, an entirely iridescent metallic blue-green starling with a striking bright orange-yellow eye",
]
CUB_CLAUDE["cape_may_warbler"] = [
    "Cape may warbler is a bird with beak that is small, slender, slightly downcurved, sharply pointed, dark gray",
    "Cape may warbler is a bird with head that is yellow face with a distinctive chestnut ear patch in breeding males, plain olive in females",
    "Cape may warbler is a bird with body that is olive-yellow above with dark streaks, bright yellow below with bold dark breast streaks",
    "Cape may warbler is a bird with wings that are dark with a bright white wing patch on the secondary coverts in males",
    "Cape may warbler is a bird with tail that is short, dark with white spots on outer feathers",
    "Cape may warbler is a bird with legs that are dark, slender, fairly short",
    "A photo of a cape may warbler, a small yellow-and-streaked warbler where breeding males show a chestnut cheek patch and bold white wing patch",
]
CUB_CLAUDE["cardinal"] = [
    "Cardinal is a bird with beak that is large, thick, conical, bright orange-red in both sexes (Northern Cardinal)",
    "Cardinal is a bird with head that is brilliant red with a tall pointed crest in males, warm tan with red highlights in females, both with a black face mask around bill",
    "Cardinal is a bird with body that is entirely brilliant scarlet-red in males, warm buff-tan with reddish tinges in females",
    "Cardinal is a bird with wings that are scarlet-red in males, tan with reddish edges in females, fairly short and rounded",
    "Cardinal is a bird with tail that is long, scarlet-red in males, tan with reddish wash in females",
    "Cardinal is a bird with legs that are pinkish-buff, fairly long, sturdy",
    "A photo of a cardinal, a familiar crested songbird where males are brilliant scarlet-red overall with a black face mask, females are warm tan with red highlights",
]
CUB_CLAUDE["carolina_wren"] = [
    "Carolina wren is a bird with beak that is long, slender, slightly downcurved, dark gray-brown",
    "Carolina wren is a bird with head that is rich rusty-brown crown with a strikingly long bold white eyebrow stripe",
    "Carolina wren is a bird with body that is warm rufous-brown above, rich buffy-cinnamon below without streaking",
    "Carolina wren is a bird with wings that are warm rufous-brown with faint dark barring",
    "Carolina wren is a bird with tail that is medium length, rufous-brown with fine dark bars, often cocked vertically",
    "Carolina wren is a bird with legs that are pinkish-buff, sturdy, fairly long",
    "A photo of a carolina wren, a warm rufous-and-buff wren with a long bold white eyebrow and a barred tail often held cocked upward",
]
CUB_CLAUDE["caspian_tern"] = [
    "Caspian tern is a bird with beak that is huge, thick, dagger-like, bright coral-red with a small dark tip",
    "Caspian tern is a bird with head that is white with a solid black cap covering crown and a slightly shaggy nape in breeding plumage, streaked black in winter",
    "Caspian tern is a bird with body that is pale gray above, white below, large and stocky for a tern",
    "Caspian tern is a bird with wings that are pale gray above with darker outer primaries, broad for a tern",
    "Caspian tern is a bird with tail that is white, only shallowly forked unlike other terns",
    "Caspian tern is a bird with legs that are short, blackish",
    "A photo of a caspian tern, a very large pale-gray tern with a massive coral-red dagger bill and a black cap",
]
CUB_CLAUDE["cedar_waxwing"] = [
    "Cedar waxwing is a bird with beak that is short, slightly hooked, dark gray-black",
    "Cedar waxwing is a bird with head that is warm cinnamon-brown with a tall pointed crest, sharp black mask through the eye, and white-edged black throat",
    "Cedar waxwing is a bird with body that is warm cinnamon-brown above blending to soft gray on the rump, pale lemon-yellow belly",
    "Cedar waxwing is a bird with wings that are gray with tiny red waxy tips on inner secondaries (the bird's namesake), no wing bars",
    "Cedar waxwing is a bird with tail that is medium, gray with a broad bright yellow tip",
    "Cedar waxwing is a bird with legs that are short, dark gray, sturdy",
    "A photo of a cedar waxwing, a sleek crested cinnamon-brown bird with a black mask, yellow belly, yellow tail tip, and red wax-like wing tips",
]
CUB_CLAUDE["cerulean_warbler"] = [
    "Cerulean warbler is a bird with beak that is small, slender, sharply pointed, dark blue-gray",
    "Cerulean warbler is a bird with head that is bright sky-cerulean-blue in males with a thin dark eye-line, blue-green with yellow eyebrow in females",
    "Cerulean warbler is a bird with body that is brilliant sky-blue above and white below in males with a thin blue chest band; females turquoise above and yellow-tinged below",
    "Cerulean warbler is a bird with wings that are blue with two crisp white wing bars in both sexes",
    "Cerulean warbler is a bird with tail that is short, blue with white spots on outer tail feathers",
    "Cerulean warbler is a bird with legs that are dark, slender, short",
    "A photo of a cerulean warbler, a tiny sky-blue-and-white warbler where males show a thin blue chest band and females are turquoise above with yellow-tinged underparts",
]
CUB_CLAUDE["chestnut_sided_warbler"] = [
    "Chestnut sided warbler is a bird with beak that is small, slender, sharply pointed, dark gray",
    "Chestnut sided warbler is a bird with head that is bright yellow crown with white face, black eye-line, and black moustache stripes in breeding males",
    "Chestnut sided warbler is a bird with body that is greenish-yellow above with black streaks, white below with bold rich chestnut streaks down the flanks",
    "Chestnut sided warbler is a bird with wings that are dark with two bold yellow-white wing bars",
    "Chestnut sided warbler is a bird with tail that is short, dark with white outer tail feather spots",
    "Chestnut sided warbler is a bird with legs that are dark, slender",
    "A photo of a chestnut sided warbler, a small warbler with bright yellow crown, white underparts, and bold rich chestnut flank streaks in breeding plumage",
]
CUB_CLAUDE["chipping_sparrow"] = [
    "Chipping sparrow is a bird with beak that is small, conical, sharply pointed, pale pinkish in winter and dark gray-black in breeding",
    "Chipping sparrow is a bird with head that is bright rufous-chestnut crown with a thin black eye-line and a clean white eyebrow stripe in breeding plumage",
    "Chipping sparrow is a bird with body that is grayish-buff above with brown streaks, plain pale grayish below without streaks",
    "Chipping sparrow is a bird with wings that are brown with two thin pale wing bars",
    "Chipping sparrow is a bird with tail that is medium, slightly notched, brown",
    "Chipping sparrow is a bird with legs that are pinkish-buff, fairly short",
    "A photo of a chipping sparrow, a small slim sparrow with a bright rufous crown, white eyebrow, and clean unstreaked underparts in breeding plumage",
]
CUB_CLAUDE["chuck_will_widow"] = [
    "Chuck will widow is a bird with beak that is tiny but with an enormous gape, mostly hidden in feathers, surrounded by long bristles",
    "Chuck will widow is a bird with head that is large, flat-topped, mottled brown with cryptic patterns, enormous dark eyes for nighttime hunting",
    "Chuck will widow is a bird with body that is intricately mottled brown, buff and rust above and below, perfect bark camouflage",
    "Chuck will widow is a bird with wings that are long, rounded, mottled brown without obvious markings, no white wing patches",
    "Chuck will widow is a bird with tail that is long, brown with white outer corners visible only on males during display",
    "Chuck will widow is a bird with legs that are very short, mostly hidden, weak, used only for perching lengthwise on branches",
    "A photo of a chuck will widow, a large cryptic mottled-brown nightjar with massive eyes, tiny bill, and bark-like camouflage plumage",
]
CUB_CLAUDE["clark_nutcracker"] = [
    "Clark nutcracker is a bird with beak that is long, sharp, pointed, all black, used for cracking pine seeds",
    "Clark nutcracker is a bird with head that is light pearly gray, blending into the body without contrasting markings",
    "Clark nutcracker is a bird with body that is uniformly light pearl-gray above and below",
    "Clark nutcracker is a bird with wings that are mostly black with a striking large white patch on the secondaries visible in flight",
    "Clark nutcracker is a bird with tail that is medium length, mostly black with bright white outer feathers, very obvious in flight",
    "Clark nutcracker is a bird with legs that are sturdy, all black",
    "A photo of a clark nutcracker, a stocky pearly-gray corvid with mostly-black wings showing a large white wing patch and white outer tail feathers",
]
CUB_CLAUDE["clay_colored_sparrow"] = [
    "Clay colored sparrow is a bird with beak that is small, conical, sharp, pinkish",
    "Clay colored sparrow is a bird with head that is buffy-brown crown with a fine dark central stripe, gray nape, distinct dark moustache stripe and pale eye-ring",
    "Clay colored sparrow is a bird with body that is sandy buff-brown above with crisp dark streaks, plain whitish-buff below without streaking",
    "Clay colored sparrow is a bird with wings that are brown with two thin pale wing bars",
    "Clay colored sparrow is a bird with tail that is medium, slightly notched, brown",
    "Clay colored sparrow is a bird with legs that are pinkish-buff, short to medium",
    "A photo of a clay colored sparrow, a pale sandy-buff sparrow with crisp face stripes, gray nape, and clean unstreaked underparts",
]
CUB_CLAUDE["cliff_swallow"] = [
    "Cliff swallow is a bird with beak that is very short, broad-based, blackish, suited for catching aerial insects",
    "Cliff swallow is a bird with head that is dark blue crown with a small white forehead patch, rusty-cinnamon throat, and pale buff collar",
    "Cliff swallow is a bird with body that is dark blue back, pale buffy-cinnamon underparts, distinctive pale buff rump",
    "Cliff swallow is a bird with wings that are dark blue, long, pointed, built for agile flight",
    "Cliff swallow is a bird with tail that is short, square-tipped, dark blue (not forked like Barn Swallow)",
    "Cliff swallow is a bird with legs that are very short, dark, weak",
    "A photo of a cliff swallow, a stocky dark-blue swallow with a square tail, white forehead patch, rusty throat, and a distinctive pale-buff rump",
]
CUB_CLAUDE["common_raven"] = [
    "Common raven is a bird with beak that is huge, thick, slightly hooked, all glossy black with prominent bristly nasal feathers",
    "Common raven is a bird with head that is uniformly glossy black with a shaggy throat ('hackles') giving a bearded appearance",
    "Common raven is a bird with body that is uniformly glossy black with subtle blue-violet iridescence overall",
    "Common raven is a bird with wings that are very long, broad, glossy black with strongly fingered primaries spread in flight",
    "Common raven is a bird with tail that is long, distinctly wedge-shaped (rather than fan-shaped like crows), all glossy black",
    "Common raven is a bird with legs that are sturdy, entirely black",
    "A photo of a common raven, a very large all-black corvid with a massive thick bill, shaggy throat hackles, and a wedge-shaped tail",
]
CUB_CLAUDE["common_tern"] = [
    "Common tern is a bird with beak that is slender, sharply pointed, orange-red with a black tip in breeding plumage, all black in winter",
    "Common tern is a bird with head that is white with a solid black cap from forehead to nape in breeding plumage, white forehead with black hindcrown in winter",
    "Common tern is a bird with body that is pale gray above, very pale gray below, white rump",
    "Common tern is a bird with wings that are pale gray, long, narrow, pointed; outer primaries gray with a darker wedge in worn plumage",
    "Common tern is a bird with tail that is deeply forked with long outer streamers, white with gray edges",
    "Common tern is a bird with legs that are short, orange-red in breeding plumage",
    "A photo of a common tern, a slender pale-gray tern with a black cap, orange-red bill with black tip, and a deeply forked tail",
]
CUB_CLAUDE["common_yellowthroat"] = [
    "Common yellowthroat is a bird with beak that is small, slender, sharply pointed, dark gray",
    "Common yellowthroat is a bird with head that is olive crown with a striking broad jet-black face mask in males bordered above by a pale gray-white stripe; plain olive in females",
    "Common yellowthroat is a bird with body that is olive-brown above, bright yellow throat and breast contrasting with whitish belly",
    "Common yellowthroat is a bird with wings that are plain olive-brown without obvious wing bars",
    "Common yellowthroat is a bird with tail that is medium, olive-brown",
    "Common yellowthroat is a bird with legs that are pinkish-buff, slender, fairly long",
    "A photo of a common yellowthroat, a small olive-and-yellow warbler where males wear a striking broad black bandit-like face mask",
]
CUB_CLAUDE["crested_auklet"] = [
    "Crested auklet is a bird with beak that is short, stout, bright orange-red in breeding plumage, with extra horny plates",
    "Crested auklet is a bird with head that is dark slate-gray with a distinctive forward-curving black crest of feathers above the bill, and white plumes behind the eye",
    "Crested auklet is a bird with body that is uniformly dark slate-gray above and below",
    "Crested auklet is a bird with wings that are dark, short, narrow, used both for flight and underwater swimming",
    "Crested auklet is a bird with tail that is short, dark, square",
    "Crested auklet is a bird with legs that are short, blackish, with webbed feet for diving",
    "A photo of a crested auklet, a small dark slate-gray seabird with an orange bill, a distinctive forward-curling crest, and a thin white plume behind each eye",
]

# ============== D ==============
CUB_CLAUDE["dark_eyed_junco"] = [
    "Dark eyed junco is a bird with beak that is short, conical, sharply pointed, pale pinkish-ivory contrasting against the dark face",
    "Dark eyed junco is a bird with head that is solid dark slate-gray hood (or rusty depending on subspecies) covering the entire head down to the chest",
    "Dark eyed junco is a bird with body that is dark slate-gray above and on the chest, sharply demarcated white belly (or rusty back depending on subspecies)",
    "Dark eyed junco is a bird with wings that are slate-gray, plain, without obvious wing bars",
    "Dark eyed junco is a bird with tail that is medium, slate-gray with conspicuous white outer tail feathers flashing in flight",
    "Dark eyed junco is a bird with legs that are pinkish-buff, fairly short",
    "A photo of a dark eyed junco, a small dark-hooded ground sparrow with a clean white belly, pale pink bill, and white-flashing outer tail feathers",
]
CUB_CLAUDE["downy_woodpecker"] = [
    "Downy woodpecker is a bird with beak that is short, small (much shorter than head), straight, chisel-tipped, blackish",
    "Downy woodpecker is a bird with head that is boldly striped black-and-white with a small red patch on the back of the head in males only",
    "Downy woodpecker is a bird with body that is white below and on the back, black wings with white spotting",
    "Downy woodpecker is a bird with wings that are black with bold white spots forming distinct rows on the flight feathers",
    "Downy woodpecker is a bird with tail that is stiff, black with white outer feathers showing small black bars",
    "Downy woodpecker is a bird with legs that are short, gray, with two toes forward and two back",
    "A photo of a downy woodpecker, a tiny black-and-white woodpecker with a small bill, white back, and a small red nape patch in males",
]

# ============== E ==============
CUB_CLAUDE["eared_grebe"] = [
    "Eared grebe is a bird with beak that is short, slender, slightly upturned, dark gray-black",
    "Eared grebe is a bird with head that is jet-black in breeding plumage with striking golden-yellow ear tufts ('ears') fanning out behind the bright red eye",
    "Eared grebe is a bird with body that is dark blackish above, rich rusty-chestnut flanks in breeding plumage, white belly",
    "Eared grebe is a bird with wings that are short, dark, mostly hidden when swimming",
    "Eared grebe is a bird with tail that is barely visible, dark, almost vestigial",
    "Eared grebe is a bird with legs that are dark, lobed (not webbed) toes set far back, suited for diving",
    "A photo of an eared grebe, a small dark grebe with bright red eyes and golden-yellow ear tufts spreading behind the eye in breeding plumage",
]
CUB_CLAUDE["eastern_towhee"] = [
    "Eastern towhee is a bird with beak that is conical, sharply pointed, all black",
    "Eastern towhee is a bird with head that is solid jet-black hood in males with red eye, dark warm-brown hood in females; both with white-marked face",
    "Eastern towhee is a bird with body that is jet-black above and on chest in males with rich rufous flanks and white belly; warm brown replaces black in females",
    "Eastern towhee is a bird with wings that are black with a distinctive white patch at the base of primaries",
    "Eastern towhee is a bird with tail that is long, black with bold white corners flashing in flight",
    "Eastern towhee is a bird with legs that are pinkish-buff, fairly long",
    "A photo of an eastern towhee, a large bicolored sparrow where males show black hood and back with rufous flanks, females replace black with warm brown",
]
CUB_CLAUDE["elegant_tern"] = [
    "Elegant tern is a bird with beak that is very long, slender, slightly drooping, bright orange-yellow",
    "Elegant tern is a bird with head that is white with a solid black cap and shaggy crest extending down the nape in breeding plumage",
    "Elegant tern is a bird with body that is pale silvery-gray above, white below, slim and elongated",
    "Elegant tern is a bird with wings that are pale gray, very long, narrow, pointed",
    "Elegant tern is a bird with tail that is deeply forked, white",
    "Elegant tern is a bird with legs that are very short, blackish",
    "A photo of an elegant tern, a slim pale-gray tern with a long thin orange-yellow drooping bill, black shaggy cap, and deeply forked white tail",
]
CUB_CLAUDE["european_goldfinch"] = [
    "European goldfinch is a bird with beak that is sharp, conical, ivory-white with a dark tip",
    "European goldfinch is a bird with head that is striking tricolor pattern with a bright red face mask, white cheeks, and a black crown extending down the nape",
    "European goldfinch is a bird with body that is warm tan-brown above, white below with a buffy chest band",
    "European goldfinch is a bird with wings that are jet-black with a brilliant broad bright-yellow wing patch and white tips on the flight feathers",
    "European goldfinch is a bird with tail that is short, black with white spots on the outer feathers",
    "European goldfinch is a bird with legs that are pale flesh-pink, fairly short",
    "A photo of a european goldfinch, a colorful small finch with a striking red face, white cheeks, black crown, and a brilliant yellow wing patch on jet-black wings",
]
CUB_CLAUDE["evening_grosbeak"] = [
    "Evening grosbeak is a bird with beak that is huge, thick, conical, pale ivory-yellow with a slight greenish wash in breeding males",
    "Evening grosbeak is a bird with head that is dark brown crown with a striking bright yellow eyebrow stripe in males, plain gray in females",
    "Evening grosbeak is a bird with body that is bright yellow body with brown shoulders in males, gray-and-yellow in females",
    "Evening grosbeak is a bird with wings that are mostly black with a large bright white patch on the secondaries",
    "Evening grosbeak is a bird with tail that is short, black, fairly square",
    "Evening grosbeak is a bird with legs that are pinkish-flesh, sturdy",
    "A photo of an evening grosbeak, a stocky finch with a massive ivory bill, where males show bright yellow body, dark head with a yellow eyebrow, and large white wing patches",
]

# ============== F ==============
CUB_CLAUDE["field_sparrow"] = [
    "Field sparrow is a bird with beak that is small, conical, sharply pointed, distinctive pinkish in all seasons",
    "Field sparrow is a bird with head that is rusty-rufous crown, gray face with a white eye-ring giving a 'wide-eyed' look, no eye-line",
    "Field sparrow is a bird with body that is grayish-rust above, plain warm buff below without streaking",
    "Field sparrow is a bird with wings that are brown with two thin whitish wing bars",
    "Field sparrow is a bird with tail that is medium, slightly notched, brown",
    "Field sparrow is a bird with legs that are pinkish, fairly short",
    "A photo of a field sparrow, a small clean-faced gray-and-rust sparrow with a pink bill, white eye-ring giving a 'wide-eyed' look, and unstreaked buffy underparts",
]
CUB_CLAUDE["fish_crow"] = [
    "Fish crow is a bird with beak that is medium-large (smaller than American Crow), slightly slimmer, all glossy black",
    "Fish crow is a bird with head that is uniformly glossy black, smooth, with thinner-looking neck profile than American Crow",
    "Fish crow is a bird with body that is uniformly glossy black with subtle iridescence",
    "Fish crow is a bird with wings that are glossy black, broad, slightly more pointed than American Crow",
    "Fish crow is a bird with tail that is medium, fan-shaped, glossy black",
    "Fish crow is a bird with legs that are sturdy, entirely black",
    "A photo of a fish crow, a slim all-black corvid resembling a smaller American Crow with a more delicate bill and head profile",
]
CUB_CLAUDE["florida_jay"] = [
    "Florida jay is a bird with beak that is medium, sharply pointed, all black",
    "Florida jay is a bird with head that is bright sky-blue crown and nape with a paler gray forehead and white eyebrow, white throat",
    "Florida jay is a bird with body that is sky-blue above, pale gray below with a distinctive blue chest band, gray belly",
    "Florida jay is a bird with wings that are bright blue without obvious bars",
    "Florida jay is a bird with tail that is long, blue, often spread when scolding",
    "Florida jay is a bird with legs that are sturdy, all black",
    "A photo of a florida jay, a slim crestless blue-and-gray jay endemic to Florida scrub with a blue hood, white eyebrow, and pale gray underparts",
]
CUB_CLAUDE["forsters_tern"] = [
    "Forsters tern is a bird with beak that is slender, pointed, orange with a black tip in breeding, all black in winter",
    "Forsters tern is a bird with head that is white with a solid black cap in breeding, distinctive isolated black ear patch (like 'mask') in winter and immatures",
    "Forsters tern is a bird with body that is pale gray above, white below, white rump",
    "Forsters tern is a bird with wings that are silvery-gray with frosty-white outer primaries (paler than other terns)",
    "Forsters tern is a bird with tail that is deeply forked, gray with white outer edges",
    "Forsters tern is a bird with legs that are short, orange-red in breeding plumage",
    "A photo of a forsters tern, a slim pale tern with frosty-white wing tips, an orange black-tipped bill, and a distinctive black ear mask in winter plumage",
]
CUB_CLAUDE["fox_sparrow"] = [
    "Fox sparrow is a bird with beak that is large for a sparrow, conical, yellowish lower mandible and dark upper mandible",
    "Fox sparrow is a bird with head that is rich rusty-rufous (eastern form) or gray-brown (western form), no obvious eye-ring",
    "Fox sparrow is a bird with body that is rusty-rufous above (eastern form), white below with bold dark triangular spots converging into a central breast spot",
    "Fox sparrow is a bird with wings that are rusty-brown with paler edges on the flight feathers",
    "Fox sparrow is a bird with tail that is medium-long, rusty-rufous, often pumped while perched",
    "Fox sparrow is a bird with legs that are pinkish-buff, sturdy, fairly long",
    "A photo of a fox sparrow, a large rusty-rufous sparrow with bold dark triangular breast spots converging into a central spot",
]
CUB_CLAUDE["frigatebird"] = [
    "Frigatebird is a bird with beak that is very long, slender, hooked, dark gray",
    "Frigatebird is a bird with head that is dark in females with white throat and chest, all-black in males with bright red inflatable throat pouch in breeding",
    "Frigatebird is a bird with body that is mostly black in males, females black above with white breast and underparts",
    "Frigatebird is a bird with wings that are extremely long, narrow, deeply angled (W-shaped silhouette), all black, built for soaring",
    "Frigatebird is a bird with tail that is very long, deeply forked (deeply scissored when closed), all black",
    "Frigatebird is a bird with legs that are very short, weak, mostly hidden, almost vestigial (cannot walk well)",
    "A photo of a frigatebird, a very large mostly-black seabird with extremely long pointed angled wings, deeply forked tail, and adult males have bright-red inflatable throat pouches",
]


# ============== G ==============
CUB_CLAUDE["gadwall"] = [
    "Gadwall is a bird with beak that is medium length, dark gray in males, orange edges with dark center in females",
    "Gadwall is a bird with head that is plain warm gray-brown in males, brown with darker eye-line in females",
    "Gadwall is a bird with body that is intricately patterned gray (vermiculated pattern of fine wavy lines) in males with a black rear, brown-mottled in females",
    "Gadwall is a bird with wings that are gray-brown with a distinctive white rectangular patch on the inner secondaries (speculum) visible in flight",
    "Gadwall is a bird with tail that is medium, gray-brown with a black undertail coverts patch in males",
    "Gadwall is a bird with legs that are yellow-orange, with fully webbed feet for swimming",
    "A photo of a gadwall, a medium gray dabbling duck where males show finely vermiculated gray plumage with a black rear and a small white wing patch",
]
CUB_CLAUDE["geococcyx"] = [
    "Geococcyx is a bird with beak that is long, sharply pointed, slightly downcurved, dark gray",
    "Geococcyx is a bird with head that is brown-streaked with a shaggy crest and a distinctive blue-and-red bare skin patch behind the eye",
    "Geococcyx is a bird with body that is heavily streaked dark brown and tan above, paler below with streaking on the chest",
    "Geococcyx is a bird with wings that are short, rounded, brown-streaked, used mostly for short flights",
    "Geococcyx is a bird with tail that is very long, brown-streaked with white tips, often cocked or fanned",
    "Geococcyx is a bird with legs that are very long, sturdy, blue-gray, built for fast running on the desert ground",
    "A photo of a geococcyx (greater roadrunner), a long-tailed streaked-brown ground cuckoo with a shaggy crest and very long legs adapted for running",
]
CUB_CLAUDE["glaucous_winged_gull"] = [
    "Glaucous winged gull is a bird with beak that is large, yellow with a red spot on the lower mandible",
    "Glaucous winged gull is a bird with head that is white in breeding adults, finely streaked dusky in winter, with a dark eye",
    "Glaucous winged gull is a bird with body that is white below, pale-medium gray mantle, large and stocky",
    "Glaucous winged gull is a bird with wings that are pale gray with distinctive gray (not black) wingtips marked with white",
    "Glaucous winged gull is a bird with tail that is white in adults, dark-banded in immatures",
    "Glaucous winged gull is a bird with legs that are pink, sturdy",
    "A photo of a glaucous winged gull, a large pale-gray gull whose wingtips are gray (not black) - matching the back color, with pink legs and a yellow bill with red spot",
]
CUB_CLAUDE["golden_winged_warbler"] = [
    "Golden winged warbler is a bird with beak that is small, slender, sharply pointed, dark gray",
    "Golden winged warbler is a bird with head that is gray with a striking bright yellow crown, broad black eye-mask and black throat in males; female with gray mask and gray throat",
    "Golden winged warbler is a bird with body that is gray above, white below with gray flanks, very clean color blocks",
    "Golden winged warbler is a bird with wings that are gray with a bold golden-yellow wing patch (covering coverts)",
    "Golden winged warbler is a bird with tail that is short, gray with white spots on outer feathers",
    "Golden winged warbler is a bird with legs that are dark, slender, short",
    "A photo of a golden winged warbler, a small gray-and-white warbler with bright yellow crown, golden wing patch, and a bold black face mask in males",
]
CUB_CLAUDE["grasshopper_sparrow"] = [
    "Grasshopper sparrow is a bird with beak that is short, conical, sharply pointed, pinkish-horn",
    "Grasshopper sparrow is a bird with head that is flat-crowned with a buffy-yellow central crown stripe and a yellow-orange spot in front of the eye, plain face",
    "Grasshopper sparrow is a bird with body that is intricately patterned brown above with chestnut and buff streaks, plain buffy below without streaks",
    "Grasshopper sparrow is a bird with wings that are short, brown with rufous edges on flight feathers",
    "Grasshopper sparrow is a bird with tail that is short, with stiff pointed tail feathers, brown",
    "Grasshopper sparrow is a bird with legs that are pinkish, fairly short",
    "A photo of a grasshopper sparrow, a small flat-headed prairie sparrow with intricate brown back streaking, plain buffy underparts, and a yellow spot near the eye",
]
CUB_CLAUDE["gray_crowned_rosy_finch"] = [
    "Gray crowned rosy finch is a bird with beak that is short, conical, dark-gray in summer and yellowish in winter",
    "Gray crowned rosy finch is a bird with head that is dark blackish-brown with a distinctive ash-gray patch covering the crown and nape",
    "Gray crowned rosy finch is a bird with body that is rich rosy-pink-flushed warm brown overall, especially on belly, flanks, and rump",
    "Gray crowned rosy finch is a bird with wings that are dark brown with rosy-pink edges on flight feathers",
    "Gray crowned rosy finch is a bird with tail that is medium, dark blackish-brown",
    "Gray crowned rosy finch is a bird with legs that are dark, sturdy, suited for foraging on alpine snowfields",
    "A photo of a gray crowned rosy finch, a chunky alpine finch with rosy-pink underparts and wings, brown body, and a distinctive ash-gray patch on the crown",
]
CUB_CLAUDE["gray_kingbird"] = [
    "Gray kingbird is a bird with beak that is large, broad, dark gray-black, slightly hooked",
    "Gray kingbird is a bird with head that is gray with a darker gray ear patch and a concealed orange crown patch",
    "Gray kingbird is a bird with body that is plain gray above, clean white below without yellow tones (unlike most other kingbirds)",
    "Gray kingbird is a bird with wings that are gray-brown with paler edges on flight feathers",
    "Gray kingbird is a bird with tail that is medium, slightly notched, gray-brown",
    "Gray kingbird is a bird with legs that are dark, fairly short",
    "A photo of a gray kingbird, a chunky gray-and-white tropical kingbird with a heavy bill, dark ear patch, and clean white underparts without yellow",
]
CUB_CLAUDE["great_crested_flycatcher"] = [
    "Great crested flycatcher is a bird with beak that is long, broad-based, all dark with a slight hook",
    "Great crested flycatcher is a bird with head that is dark gray-brown with a slightly bushy crest, gray throat",
    "Great crested flycatcher is a bird with body that is olive-brown above, gray throat and breast, and brilliant lemon-yellow belly",
    "Great crested flycatcher is a bird with wings that are dark with bold rufous-cinnamon edges on the flight feathers",
    "Great crested flycatcher is a bird with tail that is long, with conspicuous rufous-cinnamon inner webs (looks orange when fanned)",
    "Great crested flycatcher is a bird with legs that are dark, sturdy",
    "A photo of a great crested flycatcher, a large flycatcher with gray throat, bright yellow belly, slight crest, and rufous-cinnamon wings and tail",
]
CUB_CLAUDE["great_grey_shrike"] = [
    "Great grey shrike is a bird with beak that is hooked at the tip with a small tooth on the upper mandible (raptor-like), black",
    "Great grey shrike is a bird with head that is pale silvery-gray with a striking broad black mask through the eye",
    "Great grey shrike is a bird with body that is pale silvery-gray above, white below, faint barring on the breast in immatures",
    "Great grey shrike is a bird with wings that are black with a bright white patch at the base of primaries and white wing bars",
    "Great grey shrike is a bird with tail that is long, black with bright white outer tail feathers",
    "Great grey shrike is a bird with legs that are dark, sturdy, suited for perching on exposed lookouts",
    "A photo of a great grey shrike, a stocky pale-gray shrike with a broad black mask, hooked bill, white belly, and bold black-and-white wings and tail",
]
CUB_CLAUDE["green_jay"] = [
    "Green jay is a bird with beak that is medium, sharp, all black",
    "Green jay is a bird with head that is striking pattern with a bright sky-blue crown and forehead, black face mask, and white-spotted blue eyebrow",
    "Green jay is a bird with body that is bright lime-green above, lemon-yellow below",
    "Green jay is a bird with wings that are bright lime-green",
    "Green jay is a bird with tail that is long, with green central feathers and bright yellow outer feathers",
    "Green jay is a bird with legs that are dark, sturdy",
    "A photo of a green jay, a strikingly colorful jay with bright green back, yellow underparts, blue head with black mask, and a green-and-yellow tail",
]
CUB_CLAUDE["green_kingfisher"] = [
    "Green kingfisher is a bird with beak that is long, dagger-like, dark gray-black, built for spearing fish",
    "Green kingfisher is a bird with head that is glossy bottle-green crown with a slight crest, white collar around the neck",
    "Green kingfisher is a bird with body that is glossy dark green above, white below with a chestnut breast band in males or two thin green-spotted breast bands in females",
    "Green kingfisher is a bird with wings that are dark glossy green with white spots",
    "Green kingfisher is a bird with tail that is medium, dark green with white spots",
    "Green kingfisher is a bird with legs that are very short, dark, with small feet for perching",
    "A photo of a green kingfisher, a small dark-green-and-white kingfisher with a long dagger bill, white collar, and a chestnut breast band in males",
]
CUB_CLAUDE["green_tailed_towhee"] = [
    "Green tailed towhee is a bird with beak that is conical, sharp, dark with paler base",
    "Green tailed towhee is a bird with head that is bright rufous-rusty crown contrasting with gray face and bright white throat bordered by dark moustache",
    "Green tailed towhee is a bird with body that is olive-green above (only towhee with greenish back), gray below with a white throat",
    "Green tailed towhee is a bird with wings that are olive-green",
    "Green tailed towhee is a bird with tail that is long, olive-green, often flicked",
    "Green tailed towhee is a bird with legs that are pinkish-buff, fairly long",
    "A photo of a green tailed towhee, a colorful towhee with rufous crown, gray face with white throat, olive-green back and tail, and gray underparts",
]
CUB_CLAUDE["green_violetear"] = [
    "Green violetear is a bird with beak that is medium length for a hummingbird, slightly downcurved, all black",
    "Green violetear is a bird with head that is brilliant emerald-green with a striking violet-blue ear patch extending behind the eye",
    "Green violetear is a bird with body that is brilliant iridescent green above and below, with a violet-blue chest patch",
    "Green violetear is a bird with wings that are dark, narrow, pointed, beating extremely fast",
    "Green violetear is a bird with tail that is dark blue-green with a black subterminal band",
    "Green violetear is a bird with legs that are tiny, dark, used only for perching",
    "A photo of a green violetear, a brilliant iridescent green hummingbird with violet-blue ear patches and chest patch",
]
CUB_CLAUDE["gray_catbird"] = [
    "Gray catbird is a bird with beak that is short, straight, slender, entirely black with a slight downward curve at the tip",
    "Gray catbird is a bird with head that is solid slate-gray crown, distinctive black cap on the forehead and crown, dark eyes without obvious eye-ring",
    "Gray catbird is a bird with body that is uniformly dark slate-gray on upperparts and underparts, contrasting with rich rufous-chestnut undertail coverts",
    "Gray catbird is a bird with wings that are slate-gray, rounded, lacking any wing bars or pale edges",
    "Gray catbird is a bird with tail that is long, blackish-gray, often cocked or flicked sideways, contrasting with the chestnut undertail",
    "Gray catbird is a bird with legs that are blackish, long and slender, well-suited for hopping through dense thickets",
    "A photo of a gray catbird, a slim slate-gray mimid with a black cap and rusty undertail patch",
]
CUB_CLAUDE["groove_billed_ani"] = [
    "Groove billed ani is a bird with beak that is huge, deep, laterally compressed (looks like a parrot bill from the side), with grooves on the upper mandible, all black",
    "Groove billed ani is a bird with head that is glossy black with a slight crest",
    "Groove billed ani is a bird with body that is uniformly glossy black with subtle bronzy-green iridescence",
    "Groove billed ani is a bird with wings that are glossy black, fairly short and rounded",
    "Groove billed ani is a bird with tail that is very long, glossy black, often fanned or held loosely",
    "Groove billed ani is a bird with legs that are sturdy, all black, with two toes forward and two back",
    "A photo of a groove billed ani, an all-black cuckoo with a uniquely deep parrot-like ridged bill and a long loose tail",
]

# ============== H ==============
CUB_CLAUDE["harris_sparrow"] = [
    "Harris sparrow is a bird with beak that is conical, sharp, pinkish-orange",
    "Harris sparrow is a bird with head that is gray face with a striking solid black crown, throat, and chest forming a 'hood' in breeding plumage",
    "Harris sparrow is a bird with body that is rich brown-streaked above, white below with brown-streaked flanks, large for a sparrow",
    "Harris sparrow is a bird with wings that are brown with two thin white wing bars",
    "Harris sparrow is a bird with tail that is long, brown",
    "Harris sparrow is a bird with legs that are pinkish, fairly long",
    "A photo of a harris sparrow, a large sparrow with a pink bill, gray face, and a striking black hood (crown, throat, and chest) in breeding plumage",
]
CUB_CLAUDE["heermann_gull"] = [
    "Heermann gull is a bird with beak that is medium length, bright orange-red with a small black tip",
    "Heermann gull is a bird with head that is white in breeding adults, dusky in winter, with a dark eye",
    "Heermann gull is a bird with body that is uniformly dark slate-gray above and below (uniquely dark for a gull), with a paler gray belly",
    "Heermann gull is a bird with wings that are dark slate-gray with black wingtips and a thin white trailing edge",
    "Heermann gull is a bird with tail that is dark gray-black with a thin white terminal band",
    "Heermann gull is a bird with legs that are blackish, sturdy",
    "A photo of a heermann gull, a uniquely dark slate-gray gull with a bright orange-red black-tipped bill and dark legs - unmistakable among gulls",
]
CUB_CLAUDE["herring_gull"] = [
    "Herring gull is a bird with beak that is large, thick, yellow with a distinctive red spot near the tip of the lower mandible",
    "Herring gull is a bird with head that is white in breeding adults, streaked with brown in winter and immatures, pale yellow eye with orange orbital ring",
    "Herring gull is a bird with body that is white below, light pearl-gray mantle and back, juveniles mottled brown overall before maturing in 4 years",
    "Herring gull is a bird with wings that are pale gray with prominent black wingtips marked with white mirrors, broad and powerful",
    "Herring gull is a bird with tail that is white in adults, broadly banded black in immatures, square-tipped",
    "Herring gull is a bird with legs that are pink, sturdy, distinguishing it from the yellow-legged gull species",
    "A photo of a herring gull, a large pale-gray-and-white gull with pink legs, yellow bill with red spot, and black wingtips",
]
CUB_CLAUDE["henslow_sparrow"] = [
    "Henslow sparrow is a bird with beak that is large for a sparrow, conical, dark with paler base",
    "Henslow sparrow is a bird with head that is olive-green with rusty crown stripes and dark moustache and ear patch giving a striking face pattern",
    "Henslow sparrow is a bird with body that is intricately patterned with rusty wings against a buffy-streaked breast and white belly",
    "Henslow sparrow is a bird with wings that are bright rusty-rufous, very colorful for a sparrow",
    "Henslow sparrow is a bird with tail that is short, with stiff pointed feathers, brown",
    "Henslow sparrow is a bird with legs that are pinkish, fairly short",
    "A photo of a henslow sparrow, a secretive grassland sparrow with an olive-green head, distinct face stripes, bright rusty wings, and a heavily streaked breast",
]
CUB_CLAUDE["hooded_merganser"] = [
    "Hooded merganser is a bird with beak that is long, slender, slightly hooked at tip, dark in males and orange-edged in females",
    "Hooded merganser is a bird with head that is striking large fan-shaped crest, white-bordered black in males, warm rusty in females, both can raise/lower the crest dramatically",
    "Hooded merganser is a bird with body that is black above with white sides marked by black bars in males, gray-brown overall in females",
    "Hooded merganser is a bird with wings that are dark with white wing patches",
    "Hooded merganser is a bird with tail that is short, dark",
    "Hooded merganser is a bird with legs that are yellow, with fully webbed feet for diving",
    "A photo of a hooded merganser, a small slim duck where males show a striking fan-shaped white-and-black crest and dramatic black-and-white plumage; females have a rusty crest",
]
CUB_CLAUDE["hooded_oriole"] = [
    "Hooded oriole is a bird with beak that is long, slender, slightly downcurved, dark with paler base",
    "Hooded oriole is a bird with head that is bright orange-yellow in males with a black face mask and black throat (like wearing a 'hood' of orange), olive-yellow in females",
    "Hooded oriole is a bird with body that is bright orange-yellow body in males with black wings, dull olive-yellow overall in females",
    "Hooded oriole is a bird with wings that are black with a white wing bar and white edges in males",
    "Hooded oriole is a bird with tail that is long, black in males, olive in females",
    "Hooded oriole is a bird with legs that are blue-gray, fairly long",
    "A photo of a hooded oriole, a slim oriole with a downcurved bill where males show bright orange-yellow head and body, black throat, mask, wings, and tail",
]
CUB_CLAUDE["hooded_warbler"] = [
    "Hooded warbler is a bird with beak that is small, slender, sharply pointed, dark gray-black",
    "Hooded warbler is a bird with head that is bright lemon-yellow face surrounded by a striking solid black hood (covering crown, nape, and throat) in males",
    "Hooded warbler is a bird with body that is olive-green above, brilliant lemon-yellow below",
    "Hooded warbler is a bird with wings that are olive without obvious wing bars",
    "Hooded warbler is a bird with tail that is medium, often spread to flash conspicuous white outer tail feathers",
    "Hooded warbler is a bird with legs that are pinkish, slender",
    "A photo of a hooded warbler, a brilliant yellow warbler where males have a striking black 'hood' encircling a yellow face, and large white spots flash in the spread tail",
]
CUB_CLAUDE["horned_grebe"] = [
    "Horned grebe is a bird with beak that is short, slender, sharply pointed, dark with a pale tip",
    "Horned grebe is a bird with head that is striking in breeding plumage with black face, bright golden-yellow 'horns' (tufts) running from eye to behind, and rich rufous neck",
    "Horned grebe is a bird with body that is rich rufous-chestnut neck and flanks in breeding, blackish above, white belly",
    "Horned grebe is a bird with wings that are dark, short, mostly hidden when swimming",
    "Horned grebe is a bird with tail that is barely visible, almost vestigial",
    "Horned grebe is a bird with legs that are dark, lobed feet set far back, suited for diving",
    "A photo of a horned grebe, a small grebe with bright red eyes and golden-yellow 'horns' running from eye to crown in breeding plumage",
]
CUB_CLAUDE["horned_lark"] = [
    "Horned lark is a bird with beak that is short, conical, dark gray",
    "Horned lark is a bird with head that is striking with a yellow face, black mask through eye and across forehead, and small black 'horn' feather tufts on the crown",
    "Horned lark is a bird with body that is dusty-brown above, whitish below with a black chest band crescent",
    "Horned lark is a bird with wings that are brown with paler edges, fairly long and pointed",
    "Horned lark is a bird with tail that is medium, mostly black with white outer feathers",
    "Horned lark is a bird with legs that are blackish, fairly long for ground walking",
    "A photo of a horned lark, an open-country songbird with a yellow face, black mask, black chest band, and small black 'horn' feather tufts on the crown",
]
CUB_CLAUDE["horned_puffin"] = [
    "Horned puffin is a bird with beak that is huge, deep, brightly colored yellow with red tip in breeding plumage, smaller and dull in winter",
    "Horned puffin is a bird with head that is white face with black crown and a small black fleshy 'horn' above each eye in breeding plumage",
    "Horned puffin is a bird with body that is jet-black above, clean white below, sharply demarcated",
    "Horned puffin is a bird with wings that are short, narrow, dark, used both for flying and underwater swimming",
    "Horned puffin is a bird with tail that is short, dark",
    "Horned puffin is a bird with legs that are short, bright orange-red with webbed feet",
    "A photo of a horned puffin, a stocky black-and-white seabird with a huge colorful yellow-and-red bill, a white face, and small dark 'horn' projections above each eye",
]
CUB_CLAUDE["house_sparrow"] = [
    "House sparrow is a bird with beak that is short, conical, sharp, dark in breeding males and pale horn in females",
    "House sparrow is a bird with head that is gray crown with chestnut-brown sides and white cheek in males, plain dull brown in females",
    "House sparrow is a bird with body that is rich brown back with black streaks and a black throat-bib in males; plain dull buffy-brown in females",
    "House sparrow is a bird with wings that are brown with a single small white wing bar",
    "House sparrow is a bird with tail that is medium, dark brown",
    "House sparrow is a bird with legs that are pale pinkish-buff, short",
    "A photo of a house sparrow, a familiar urban sparrow where males show gray crown, chestnut sides, white cheek, and a black bib on the throat; females are plain brown",
]
CUB_CLAUDE["house_wren"] = [
    "House wren is a bird with beak that is long, slender, slightly downcurved, dark gray-brown",
    "House wren is a bird with head that is plain warm brown without obvious eyebrow or eye-ring",
    "House wren is a bird with body that is uniformly warm brown above with very faint barring, paler grayish-brown below",
    "House wren is a bird with wings that are warm brown with very fine dark barring",
    "House wren is a bird with tail that is medium length, often cocked vertically, brown with fine dark barring",
    "House wren is a bird with legs that are pinkish-buff, fairly long, suited for hopping",
    "A photo of a house wren, a small plain warm-brown wren with very faint markings, often holding its tail cocked upward",
]

# ============== I ==============
CUB_CLAUDE["indigo_bunting"] = [
    "Indigo bunting is a bird with beak that is short, conical, sharply pointed, silvery-gray",
    "Indigo bunting is a bird with head that is brilliant deep indigo-blue all over in adult breeding males, plain warm brown in females",
    "Indigo bunting is a bird with body that is uniformly brilliant indigo-blue all over in breeding males, plain warm buffy-brown in females",
    "Indigo bunting is a bird with wings that are blackish-blue in males, brown with two faint buff wing bars in females",
    "Indigo bunting is a bird with tail that is medium, dark blue in males, brown in females",
    "Indigo bunting is a bird with legs that are dark gray-black, slender",
    "A photo of an indigo bunting, a tiny finch-like songbird where breeding males are brilliant deep blue all over and females are plain warm brown",
]
CUB_CLAUDE["ivory_gull"] = [
    "Ivory gull is a bird with beak that is medium, blue-gray base with bright yellow tip",
    "Ivory gull is a bird with head that is entirely pure white, dark eye contrasting with white face",
    "Ivory gull is a bird with body that is uniformly pure white above and below in adults, juveniles dingy with dark spotting",
    "Ivory gull is a bird with wings that are entirely pure white with no contrasting markings (unique among gulls)",
    "Ivory gull is a bird with tail that is pure white in adults",
    "Ivory gull is a bird with legs that are short, blackish, sturdy",
    "A photo of an ivory gull, an Arctic gull with entirely pure-white plumage, black legs, and a blue-and-yellow bill - unmistakable among gulls",
]

# ============== K ==============
CUB_CLAUDE["kentucky_warbler"] = [
    "Kentucky warbler is a bird with beak that is small, slender, sharply pointed, dark gray-black",
    "Kentucky warbler is a bird with head that is olive crown with a striking bright yellow eyebrow that wraps around to form 'spectacles', and a broad black face mask in males",
    "Kentucky warbler is a bird with body that is olive-green above, brilliant lemon-yellow below",
    "Kentucky warbler is a bird with wings that are plain olive without wing bars",
    "Kentucky warbler is a bird with tail that is short, plain olive without conspicuous markings",
    "Kentucky warbler is a bird with legs that are pinkish, sturdy, fairly long",
    "A photo of a kentucky warbler, a yellow-and-olive ground warbler with bright yellow spectacles and a broad black face mask in males",
]

# ============== L ==============
CUB_CLAUDE["laysan_albatross"] = [
    "Laysan albatross is a bird with beak that is huge, long, hooked, pinkish with darker tip, built for tearing food at sea",
    "Laysan albatross is a bird with head that is mostly white with a dark smudge around and behind the eye, dark partial eye-mask",
    "Laysan albatross is a bird with body that is white below and on the head, dark gray-brown above (mantle and back)",
    "Laysan albatross is a bird with wings that are extremely long, narrow, pointed, dark gray-brown above and white-marked below, built for dynamic soaring",
    "Laysan albatross is a bird with tail that is short, dark gray-brown",
    "Laysan albatross is a bird with legs that are pinkish-flesh, with webbed feet for swimming",
    "A photo of a laysan albatross, a large pelagic seabird with white head and underparts, dark eye smudge, dark gray-brown back and wings, and a pink hooked bill",
]
CUB_CLAUDE["lazuli_bunting"] = [
    "Lazuli bunting is a bird with beak that is short, conical, silvery-gray",
    "Lazuli bunting is a bird with head that is brilliant turquoise-blue head and back in males, plain warm gray-buff in females",
    "Lazuli bunting is a bird with body that is brilliant turquoise-blue above with a rich orange-cinnamon chest, and white belly in males; plain buff with hint of blue in wings in females",
    "Lazuli bunting is a bird with wings that are blue with two crisp white wing bars in males, brown with buffy wing bars in females",
    "Lazuli bunting is a bird with tail that is medium, dark blue in males, brown in females",
    "Lazuli bunting is a bird with legs that are dark, slender",
    "A photo of a lazuli bunting, a tiny finch-like songbird where males show brilliant turquoise-blue head and back, orange-cinnamon chest, white belly, and white wing bars",
]
CUB_CLAUDE["le_conte_sparrow"] = [
    "Le conte sparrow is a bird with beak that is small, conical, dark with paler base",
    "Le conte sparrow is a bird with head that is striking burnt-orange face, gray ear patch, and a buffy-yellow central crown stripe",
    "Le conte sparrow is a bird with body that is intricately streaked with chestnut, white, and black on the back, buffy-orange chest with fine dark streaks on flanks",
    "Le conte sparrow is a bird with wings that are brown with rufous edges, no wing bars",
    "Le conte sparrow is a bird with tail that is short, with stiff pointed feathers, brown",
    "Le conte sparrow is a bird with legs that are pinkish, short",
    "A photo of a le conte sparrow, a tiny secretive sparrow with a striking burnt-orange face, gray ear patch, and intricately streaked back",
]
CUB_CLAUDE["least_auklet"] = [
    "Least auklet is a bird with beak that is very short, stubby, dark with bright red tip in breeding plumage",
    "Least auklet is a bird with head that is dark slate-gray with white forehead plumes and a small white plume behind eye in breeding",
    "Least auklet is a bird with body that is dark slate-gray above, white below with variable gray spotting on the breast",
    "Least auklet is a bird with wings that are short, narrow, dark, used for both flying and underwater swimming",
    "Least auklet is a bird with tail that is short, dark",
    "Least auklet is a bird with legs that are short, dark gray, with webbed feet",
    "A photo of a least auklet, a tiny dark-and-white seabird with a stubby red-tipped bill and white facial plumes in breeding plumage",
]
CUB_CLAUDE["least_flycatcher"] = [
    "Least flycatcher is a bird with beak that is small, broad-based, dark upper and pale orange-pink lower mandible",
    "Least flycatcher is a bird with head that is grayish-olive with a bold white eye-ring (rounded, complete), no crest",
    "Least flycatcher is a bird with body that is grayish-olive above, pale grayish-white below with a faint olive-tan wash across the chest",
    "Least flycatcher is a bird with wings that are dark with two clean white wing bars",
    "Least flycatcher is a bird with tail that is medium, dark olive-brown, often flicked",
    "Least flycatcher is a bird with legs that are dark, slender",
    "A photo of a least flycatcher, a small olive-and-white empidonax flycatcher with a bold complete white eye-ring and two clean wing bars",
]
CUB_CLAUDE["least_tern"] = [
    "Least tern is a bird with beak that is slender, sharply pointed, yellow with a black tip in breeding plumage",
    "Least tern is a bird with head that is white with a black cap and a distinctive white forehead patch in breeding plumage",
    "Least tern is a bird with body that is pale gray above, white below, smallest of the terns",
    "Least tern is a bird with wings that are pale gray with a black leading edge on the outer primaries, very narrow and pointed",
    "Least tern is a bird with tail that is forked, white",
    "Least tern is a bird with legs that are short, yellow-orange",
    "A photo of a least tern, the smallest tern, with a yellow black-tipped bill, white forehead patch within the black cap, and pale gray wings with black leading edge",
]
CUB_CLAUDE["lincoln_sparrow"] = [
    "Lincoln sparrow is a bird with beak that is small, conical, sharp, dark with paler base",
    "Lincoln sparrow is a bird with head that is gray with a rusty crown, fine dark streaks, and a buffy submoustache patch contrasting with white throat",
    "Lincoln sparrow is a bird with body that is gray-brown above with fine streaks, distinctive buffy band across upper breast covered with very fine dark streaks (looks finely 'penciled'), white belly",
    "Lincoln sparrow is a bird with wings that are brown with two thin pale wing bars",
    "Lincoln sparrow is a bird with tail that is medium, slightly notched, brown",
    "Lincoln sparrow is a bird with legs that are pinkish, slender",
    "A photo of a lincoln sparrow, a delicate sparrow with a gray face, rusty crown, and a buffy upper breast band crossed with extremely fine dark pencil streaks",
]
CUB_CLAUDE["loggerhead_shrike"] = [
    "Loggerhead shrike is a bird with beak that is heavy, hooked at the tip with a small tooth (raptor-like), all black",
    "Loggerhead shrike is a bird with head that is gray with a strikingly broad black mask through the eye extending across forehead",
    "Loggerhead shrike is a bird with body that is gray above, white below with very faint barring",
    "Loggerhead shrike is a bird with wings that are black with a bold white patch at the base of primaries",
    "Loggerhead shrike is a bird with tail that is medium, black with white outer feathers",
    "Loggerhead shrike is a bird with legs that are dark, sturdy",
    "A photo of a loggerhead shrike, a stocky predatory songbird with gray-and-white plumage, a strikingly broad black face mask, and bold black-and-white wings and tail",
]
CUB_CLAUDE["long_tailed_jaeger"] = [
    "Long tailed jaeger is a bird with beak that is medium, hooked, dark gray-black",
    "Long tailed jaeger is a bird with head that is white in adults with a sharply demarcated black cap, dark in juveniles",
    "Long tailed jaeger is a bird with body that is pale gray above, white below in breeding adults, brown-banded in juveniles",
    "Long tailed jaeger is a bird with wings that are gray-brown, long, narrow, pointed, with only minimal white at primary bases",
    "Long tailed jaeger is a bird with tail that is distinctly long, with extremely elongated central feathers (longer streamers than other jaegers) in adults",
    "Long tailed jaeger is a bird with legs that are short, dark blackish",
    "A photo of a long tailed jaeger, a slim graceful seabird with extremely long central tail streamers, gray-and-white plumage, and a black cap in breeding adults",
]
CUB_CLAUDE["louisiana_waterthrush"] = [
    "Louisiana waterthrush is a bird with beak that is medium, slender, sharply pointed, dark with paler base",
    "Louisiana waterthrush is a bird with head that is olive-brown with a striking bold white eyebrow that flares wider behind the eye, dark crown",
    "Louisiana waterthrush is a bird with body that is olive-brown above, white below with bold dark streaks on chest and flanks, pale buffy throat (no streaks)",
    "Louisiana waterthrush is a bird with wings that are plain olive-brown without wing bars",
    "Louisiana waterthrush is a bird with tail that is medium, often pumped up and down constantly, plain brown",
    "Louisiana waterthrush is a bird with legs that are pale pinkish-flesh, fairly long",
    "A photo of a louisiana waterthrush, a ground warbler with olive-brown back, white belly with bold dark streaks, a wide white flaring eyebrow, and a constantly bobbing tail",
]


# ============== M ==============
CUB_CLAUDE["magnolia_warbler"] = [
    "Magnolia warbler is a bird with beak that is small, slender, sharply pointed, all black",
    "Magnolia warbler is a bird with head that is gray crown with a striking white eyebrow, jet-black face mask, and white arc under the eye",
    "Magnolia warbler is a bird with body that is gray-and-black above with bold black streaks, brilliant lemon-yellow below with heavy bold black streaks down the flanks",
    "Magnolia warbler is a bird with wings that are black with two bold large white wing bars (almost forming a single white patch in breeding males)",
    "Magnolia warbler is a bird with tail that is unique - black with a wide white band across the middle visible from below ('bandit's tail')",
    "Magnolia warbler is a bird with legs that are dark, slender, short",
    "A photo of a magnolia warbler, a striking small warbler with bold yellow underparts, heavy black flank streaks, black mask, large white wing patch, and a unique white-banded tail",
]
CUB_CLAUDE["mallard"] = [
    "Mallard is a bird with beak that is broad, flat (typical duck shape), bright yellow in males, orange with darker patches in females",
    "Mallard is a bird with head that is iridescent emerald-green with a thin white neck-ring in males, plain mottled brown with dark eye-line in females",
    "Mallard is a bird with body that is gray-and-chestnut above with a chestnut breast in males, mottled brown overall in females",
    "Mallard is a bird with wings that are gray-brown with a vivid blue speculum bordered by black-and-white bars",
    "Mallard is a bird with tail that is white with a black curl at the rump in males, plain brown in females",
    "Mallard is a bird with legs that are bright orange, with fully webbed feet for swimming",
    "A photo of a mallard, a familiar dabbling duck where males show iridescent green head, white collar, chestnut breast, and gray body; females are mottled brown",
]
CUB_CLAUDE["mangrove_cuckoo"] = [
    "Mangrove cuckoo is a bird with beak that is long, slightly downcurved, dark with yellow lower mandible",
    "Mangrove cuckoo is a bird with head that is plain gray-brown crown with a black ear patch (mask) extending behind the eye",
    "Mangrove cuckoo is a bird with body that is plain warm gray-brown above, rich buff-cinnamon underparts (deeper than other cuckoos)",
    "Mangrove cuckoo is a bird with wings that are gray-brown without rufous primaries",
    "Mangrove cuckoo is a bird with tail that is very long, brown with bold white spots on the tips of the outer feathers",
    "Mangrove cuckoo is a bird with legs that are short, gray, with two toes forward and two back",
    "A photo of a mangrove cuckoo, a slim long-tailed cuckoo with gray-brown upperparts, rich buff underparts, a black ear patch, and a yellow lower mandible",
]
CUB_CLAUDE["marsh_wren"] = [
    "Marsh wren is a bird with beak that is long, slender, slightly downcurved, dark with paler base",
    "Marsh wren is a bird with head that is rusty-brown crown with a striking bold white eyebrow stripe and a brown ear patch",
    "Marsh wren is a bird with body that is warm rusty-brown above with bold black-and-white striping on the back, plain pale buff-cinnamon below",
    "Marsh wren is a bird with wings that are warm brown with fine black barring",
    "Marsh wren is a bird with tail that is medium, often cocked vertically, brown with dark barring",
    "Marsh wren is a bird with legs that are pinkish-buff, fairly long, suited for clinging to reed stems",
    "A photo of a marsh wren, a small rusty-brown wren with a bold white eyebrow and a strikingly black-and-white striped back, often holding tail cocked",
]
CUB_CLAUDE["mockingbird"] = [
    "Mockingbird is a bird with beak that is long, slender, slightly downcurved, dark gray-black",
    "Mockingbird is a bird with head that is plain gray with a thin dark eye-line, bright yellow eye",
    "Mockingbird is a bird with body that is plain medium gray above, plain pale gray below",
    "Mockingbird is a bird with wings that are gray with bold large white wing patches at the base of primaries (very visible in flight)",
    "Mockingbird is a bird with tail that is long, dark gray with white outer feathers (also very visible in flight)",
    "Mockingbird is a bird with legs that are dark, fairly long, suited for hopping on lawns",
    "A photo of a mockingbird, a slim gray-and-pale-gray songbird with a long tail, yellow eye, and bold white wing patches and outer tail feathers flashing in flight",
]
CUB_CLAUDE["mourning_warbler"] = [
    "Mourning warbler is a bird with beak that is small, slender, sharply pointed, dark gray-black",
    "Mourning warbler is a bird with head that is olive crown with a slate-gray hood (covering crown, nape, throat) and a black 'crepe' patch on the chest in breeding males",
    "Mourning warbler is a bird with body that is olive-green above, brilliant lemon-yellow below (gray hood meets yellow chest sharply)",
    "Mourning warbler is a bird with wings that are plain olive without obvious wing bars",
    "Mourning warbler is a bird with tail that is short, plain olive",
    "Mourning warbler is a bird with legs that are pinkish, sturdy",
    "A photo of a mourning warbler, a small ground warbler with a slate-gray hood, black crepe-like chest patch in males, olive back, and bright yellow underparts",
]
CUB_CLAUDE["myrtle_warbler"] = [
    "Myrtle warbler is a bird with beak that is small, slender, sharply pointed, dark",
    "Myrtle warbler is a bird with head that is gray with a yellow crown patch, white throat (key distinction from Audubon's), and dark face mask in males",
    "Myrtle warbler is a bird with body that is gray-blue above with black streaks, white below with black breast and bold yellow side patches",
    "Myrtle warbler is a bird with wings that are dark with two clean white wing bars",
    "Myrtle warbler is a bird with tail that is short, dark with white spots on outer tail feathers, distinctive bright yellow rump patch above",
    "Myrtle warbler is a bird with legs that are dark, slender",
    "A photo of a myrtle warbler, a small gray-blue warbler with white throat, yellow crown, yellow side patches, yellow rump, and black breast streaks in breeding males",
]

# ============== N ==============
CUB_CLAUDE["nashville_warbler"] = [
    "Nashville warbler is a bird with beak that is small, slender, sharply pointed, dark gray-black",
    "Nashville warbler is a bird with head that is gray crown with a complete bold white eye-ring (no eyebrow), with a small concealed rufous crown patch in males",
    "Nashville warbler is a bird with body that is olive-green above, brilliant lemon-yellow below including chin and undertail coverts",
    "Nashville warbler is a bird with wings that are plain olive without obvious wing bars",
    "Nashville warbler is a bird with tail that is short, plain olive",
    "Nashville warbler is a bird with legs that are dark, slender",
    "A photo of a nashville warbler, a small warbler with a gray hood, complete white eye-ring (no eyebrow), olive back, and brilliant yellow underparts including the throat",
]
CUB_CLAUDE["nelson_sharp_tailed_sparrow"] = [
    "Nelson sharp tailed sparrow is a bird with beak that is small, conical, sharp, dark with pale base",
    "Nelson sharp tailed sparrow is a bird with head that is striking with a buffy-orange face surrounded by a gray central crown stripe and a gray ear patch",
    "Nelson sharp tailed sparrow is a bird with body that is buffy-streaked above, buffy-orange chest with very fine streaks, white belly",
    "Nelson sharp tailed sparrow is a bird with wings that are brown with rufous edges, no obvious wing bars",
    "Nelson sharp tailed sparrow is a bird with tail that is short, with stiff pointed feathers (sharp-tailed)",
    "Nelson sharp tailed sparrow is a bird with legs that are pinkish-buff, short",
    "A photo of a nelson sharp tailed sparrow, a small marsh sparrow with a striking buffy-orange face encircled by gray, and stiff pointed tail feathers",
]
CUB_CLAUDE["nighthawk"] = [
    "Nighthawk is a bird with beak that is tiny but with a huge gape lined with bristles",
    "Nighthawk is a bird with head that is large, flat-topped, mottled cryptic gray-and-brown, with very large dark eyes",
    "Nighthawk is a bird with body that is intricately patterned mottled gray, brown, and buff above and below for camouflage on bare ground",
    "Nighthawk is a bird with wings that are very long, narrow, pointed, mottled gray-brown with a distinctive bold white wing bar across the primaries (visible in flight)",
    "Nighthawk is a bird with tail that is long, mottled brown with a white subterminal band in males",
    "Nighthawk is a bird with legs that are very short, weak, mostly hidden, used only for resting on perches",
    "A photo of a nighthawk, a slim aerial nightjar with very long pointed wings showing bold white wing bars in flight, cryptic mottled gray-brown plumage",
]
CUB_CLAUDE["northern_flicker"] = [
    "Northern flicker is a bird with beak that is medium-long, slightly curved (more than other woodpeckers), gray-black",
    "Northern flicker is a bird with head that is gray crown with a red nape crescent (eastern Yellow-shafted) or red moustache (western Red-shafted), brown face",
    "Northern flicker is a bird with body that is brown-barred back, white belly with bold black spots, distinctive black crescent on the chest",
    "Northern flicker is a bird with wings that are brown with bold black bars; bright yellow flight feather shafts (Yellow-shafted) or salmon-pink shafts (Red-shafted) flash in flight",
    "Northern flicker is a bird with tail that is medium, brown-and-black-barred, with bright white rump conspicuous in flight",
    "Northern flicker is a bird with legs that are gray, with two toes forward and two back",
    "A photo of a northern flicker, a brown woodpecker with black-spotted underparts, black chest crescent, white rump, and yellow (or red) underwing flashes",
]
CUB_CLAUDE["northern_fulmar"] = [
    "Northern fulmar is a bird with beak that is short, thick, hooked, with prominent tubular nostrils on top, yellow with darker tip",
    "Northern fulmar is a bird with head that is white in light morph, all gray in dark morph, with a thick neck and a stout 'bull-headed' look",
    "Northern fulmar is a bird with body that is pale gray (light morph) above and white below, or all uniformly gray (dark morph) overall",
    "Northern fulmar is a bird with wings that are stiff, narrow, gray, often held straight in 'stiff-winged' soaring flight",
    "Northern fulmar is a bird with tail that is short, gray",
    "Northern fulmar is a bird with legs that are pinkish-flesh, with webbed feet",
    "A photo of a northern fulmar, a stocky gull-like seabird with a thick neck, tube-nosed yellow bill, and stiff-winged soaring flight - light morph or all-gray dark morph",
]
CUB_CLAUDE["northern_waterthrush"] = [
    "Northern waterthrush is a bird with beak that is medium, slender, pointed, dark with paler base",
    "Northern waterthrush is a bird with head that is olive-brown with a long buffy eyebrow stripe of even width (does not flare behind eye, unlike Louisiana)",
    "Northern waterthrush is a bird with body that is olive-brown above, pale yellowish-buff below with bold dark streaks on throat, chest and flanks (throat is also streaked, unlike Louisiana)",
    "Northern waterthrush is a bird with wings that are plain olive-brown without wing bars",
    "Northern waterthrush is a bird with tail that is medium, often pumped up and down constantly",
    "Northern waterthrush is a bird with legs that are pinkish-flesh, fairly long",
    "A photo of a northern waterthrush, a ground warbler similar to Louisiana but with even-width buffy eyebrow, streaked throat, and yellower buff underparts",
]

# ============== O ==============
CUB_CLAUDE["olive_sided_flycatcher"] = [
    "Olive sided flycatcher is a bird with beak that is large, broad, dark with paler lower mandible",
    "Olive sided flycatcher is a bird with head that is large, big-headed, plain dark olive-gray with no obvious crest",
    "Olive sided flycatcher is a bird with body that is dark olive-gray with a distinctive 'open vest' look - dark sides contrasting with a white center stripe down the chest",
    "Olive sided flycatcher is a bird with wings that are dark olive-gray, long, pointed",
    "Olive sided flycatcher is a bird with tail that is medium, dark olive-gray",
    "Olive sided flycatcher is a bird with legs that are dark, sturdy",
    "A photo of an olive sided flycatcher, a stocky big-headed flycatcher with dark olive-gray sides framing a white central chest stripe (looks like wearing an open vest)",
]
CUB_CLAUDE["orange_crowned_warbler"] = [
    "Orange crowned warbler is a bird with beak that is small, slender, sharply pointed, dark gray",
    "Orange crowned warbler is a bird with head that is plain olive-yellow with a faint pale eyebrow and dark eye-line, concealed orange crown patch rarely visible",
    "Orange crowned warbler is a bird with body that is plain olive-yellow above, pale yellowish-olive below with very faint blurry breast streaks",
    "Orange crowned warbler is a bird with wings that are plain olive without wing bars",
    "Orange crowned warbler is a bird with tail that is short, plain olive",
    "Orange crowned warbler is a bird with legs that are dark, slender",
    "A photo of an orange crowned warbler, a small drab uniformly olive-yellow warbler with no wing bars, faint streaking, and a usually concealed orange crown patch",
]
CUB_CLAUDE["orchard_oriole"] = [
    "Orchard oriole is a bird with beak that is slender, sharply pointed, slightly downcurved, dark gray with paler base",
    "Orchard oriole is a bird with head that is solid jet-black hood (covering head, throat, upper back) in adult males, plain olive-yellow in females and immatures",
    "Orchard oriole is a bird with body that is rich brick-chestnut below in males with black above, plain olive-yellow overall in females",
    "Orchard oriole is a bird with wings that are black with a chestnut shoulder patch and white wing bar in males, olive with two pale wing bars in females",
    "Orchard oriole is a bird with tail that is medium, all black in males, olive in females",
    "Orchard oriole is a bird with legs that are blue-gray, fairly long",
    "A photo of an orchard oriole, the smallest oriole, where adult males show black head and back with rich brick-chestnut underparts; females are uniformly olive-yellow",
]
CUB_CLAUDE["ovenbird"] = [
    "Ovenbird is a bird with beak that is medium, slender, sharply pointed, dark with pale lower mandible",
    "Ovenbird is a bird with head that is olive crown with two distinctive black side stripes flanking a rusty-orange central crown stripe",
    "Ovenbird is a bird with body that is plain olive-brown above, white below with bold black streaks on the chest and flanks",
    "Ovenbird is a bird with wings that are plain olive-brown without wing bars",
    "Ovenbird is a bird with tail that is short, plain olive-brown",
    "Ovenbird is a bird with legs that are pinkish-flesh, fairly long, suited for walking on forest floor",
    "A photo of an ovenbird, a ground-walking warbler with olive upperparts, white underparts boldly streaked black, and a striking orange-and-black central crown stripe",
]

# ============== P ==============
CUB_CLAUDE["pacific_loon"] = [
    "Pacific loon is a bird with beak that is straight, sharp, slender, dark gray (held horizontally, unlike Common Loon)",
    "Pacific loon is a bird with head that is smooth pale silvery-gray crown extending down nape, glossy black throat with thin black-and-white striped patch in breeding plumage",
    "Pacific loon is a bird with body that is glossy black above with neat rows of white square spots, white below in breeding; gray-brown overall in winter",
    "Pacific loon is a bird with wings that are blackish above with white spotting, dark below",
    "Pacific loon is a bird with tail that is short, dark",
    "Pacific loon is a bird with legs that are dark, set far back, with fully webbed feet for diving",
    "A photo of a pacific loon, a slim diving bird with smooth silver-gray crown, black throat, and neat rows of white square spots on a glossy black back in breeding plumage",
]
CUB_CLAUDE["painted_bunting"] = [
    "Painted bunting is a bird with beak that is short, conical, silvery-gray",
    "Painted bunting is a bird with head that is electric cobalt-blue head and nape in adult males, plain bright lime-green head in females",
    "Painted bunting is a bird with body that is brilliant tricolor in males - blue head, lime-green back, and crimson-red underparts; uniformly bright lime-green overall in females",
    "Painted bunting is a bird with wings that are dark with green-and-red highlights in males, plain green in females",
    "Painted bunting is a bird with tail that is medium, dark in males, green in females",
    "Painted bunting is a bird with legs that are dark, slender",
    "A photo of a painted bunting, a tiny finch where adult males show brilliant tricolor blue head, lime-green back, and crimson-red underparts; females are uniformly lime-green",
]
CUB_CLAUDE["palm_warbler"] = [
    "Palm warbler is a bird with beak that is small, slender, sharply pointed, dark",
    "Palm warbler is a bird with head that is rusty-chestnut crown in breeding plumage with a yellow eyebrow, dull brown crown in winter",
    "Palm warbler is a bird with body that is olive-brown above, yellow-tinged below with chestnut streaks in breeding (Eastern form is brighter yellow), with a distinctive yellow rump and undertail coverts",
    "Palm warbler is a bird with wings that are dark olive-brown without obvious wing bars",
    "Palm warbler is a bird with tail that is short, often pumped up and down constantly, brown with white spots on outer feathers",
    "Palm warbler is a bird with legs that are pinkish, fairly long",
    "A photo of a palm warbler, a ground-foraging warbler with rusty crown in breeding, yellow eyebrow, yellow undertail, and a constantly bobbing tail",
]
CUB_CLAUDE["parakeet_auklet"] = [
    "Parakeet auklet is a bird with beak that is short, stout, distinctively curved upward (parrot-like), bright orange-red in breeding plumage",
    "Parakeet auklet is a bird with head that is dark slate-gray with a thin white plume behind each eye in breeding, white throat",
    "Parakeet auklet is a bird with body that is dark slate-gray above, white below with sharp demarcation",
    "Parakeet auklet is a bird with wings that are short, narrow, dark, used both for flight and underwater swimming",
    "Parakeet auklet is a bird with tail that is short, dark",
    "Parakeet auklet is a bird with legs that are short, blackish, with webbed feet",
    "A photo of a parakeet auklet, a small dark-and-white seabird with a distinctively upturned orange-red bill (parrot-shaped) and a thin white plume behind each eye",
]
CUB_CLAUDE["pelagic_cormorant"] = [
    "Pelagic cormorant is a bird with beak that is slender, hooked, all dark gray-black",
    "Pelagic cormorant is a bird with head that is glossy iridescent purple-green-black with two thin filamentary crests on the crown in breeding plumage, and a small red patch at the base of the bill",
    "Pelagic cormorant is a bird with body that is uniformly slim glossy iridescent dark purple-green-black overall with a striking white flank patch in breeding plumage",
    "Pelagic cormorant is a bird with wings that are dark, fairly narrow",
    "Pelagic cormorant is a bird with tail that is medium, stiff, dark",
    "Pelagic cormorant is a bird with legs that are short, dark, with fully webbed feet for diving",
    "A photo of a pelagic cormorant, the slimmest cormorant, with glossy purple-green-black plumage, two thin breeding crests, and a bright white flank patch in breeding",
]
CUB_CLAUDE["philadelphia_vireo"] = [
    "Philadelphia vireo is a bird with beak that is short, thick, slightly hooked, dark gray",
    "Philadelphia vireo is a bird with head that is gray crown with a thin white eyebrow and a thin dark eye-line",
    "Philadelphia vireo is a bird with body that is plain gray-olive above, distinctive bright yellow on the throat and upper breast (key field mark, no other Empidonax-style vireo has this)",
    "Philadelphia vireo is a bird with wings that are plain olive-gray without wing bars",
    "Philadelphia vireo is a bird with tail that is short, plain olive-gray",
    "Philadelphia vireo is a bird with legs that are dark, sturdy",
    "A photo of a philadelphia vireo, a small olive-gray vireo with no wing bars, distinguished by its strongly yellow-tinged throat and upper breast",
]
CUB_CLAUDE["pied_billed_grebe"] = [
    "Pied billed grebe is a bird with beak that is short, thick, almost chicken-like, pale ivory with a black ring around it in breeding plumage",
    "Pied billed grebe is a bird with head that is plain warm brown with a black throat patch in breeding plumage",
    "Pied billed grebe is a bird with body that is plain warm brown above, paler buff-brown below, no contrasting markings",
    "Pied billed grebe is a bird with wings that are short, brown, mostly hidden when swimming",
    "Pied billed grebe is a bird with tail that is barely visible, almost vestigial",
    "Pied billed grebe is a bird with legs that are dark, lobed feet set far back, suited for diving",
    "A photo of a pied billed grebe, a stocky brown grebe with a chicken-like ivory bill marked by a black ring and a black throat patch in breeding plumage",
]
CUB_CLAUDE["pied_kingfisher"] = [
    "Pied kingfisher is a bird with beak that is long, dagger-like, all black, built for spearing fish",
    "Pied kingfisher is a bird with head that is striking pied (black-and-white) pattern with a black crown, white face, and a slight crest",
    "Pied kingfisher is a bird with body that is sharply patterned black-and-white above and below, with one black breast band in females and two black breast bands in males",
    "Pied kingfisher is a bird with wings that are black with bold white spots and bars",
    "Pied kingfisher is a bird with tail that is medium, black-and-white-barred",
    "Pied kingfisher is a bird with legs that are short, blackish",
    "A photo of a pied kingfisher, a striking black-and-white kingfisher with a long dagger bill, slight crest, and bold pied markings; males have two breast bands and females one",
]
CUB_CLAUDE["pigeon_guillemot"] = [
    "Pigeon guillemot is a bird with beak that is medium, slender, sharply pointed, all black with bright red gape inside",
    "Pigeon guillemot is a bird with head that is glossy black overall in breeding plumage, mottled white-and-gray in winter",
    "Pigeon guillemot is a bird with body that is uniformly glossy black above and below in breeding, mostly white with mottled-gray back in winter",
    "Pigeon guillemot is a bird with wings that are black with a striking large oval white wing patch on the inner secondaries, conspicuous in flight",
    "Pigeon guillemot is a bird with tail that is short, black",
    "Pigeon guillemot is a bird with legs that are bright red-orange, with webbed feet",
    "A photo of a pigeon guillemot, a black seabird with a striking large oval white wing patch and bright red feet visible against the all-black breeding plumage",
]
CUB_CLAUDE["pileated_woodpecker"] = [
    "Pileated woodpecker is a bird with beak that is very large, long, chisel-tipped, dark blackish",
    "Pileated woodpecker is a bird with head that is striking with a tall flaming-red flat crest, white-and-black face stripes, and red moustache stripe in males (black moustache in females)",
    "Pileated woodpecker is a bird with body that is mostly black with a white throat and white wing-linings flashing in flight, large crow-sized",
    "Pileated woodpecker is a bird with wings that are black with bright white wing-linings (very visible in flight)",
    "Pileated woodpecker is a bird with tail that is long, stiff, all black, used as a brace",
    "Pileated woodpecker is a bird with legs that are gray, with two toes forward and two back",
    "A photo of a pileated woodpecker, a very large mostly-black woodpecker with a tall flaming-red crest, bold white face stripes, and brilliant white wing-linings flashing in flight",
]
CUB_CLAUDE["pine_grosbeak"] = [
    "Pine grosbeak is a bird with beak that is large, thick, conical, slightly hooked, dark gray-black",
    "Pine grosbeak is a bird with head that is rich rosy-red in adult males, warm yellowish-orange to golden-olive in females and immatures",
    "Pine grosbeak is a bird with body that is rosy-red overall in males, gray with yellowish-orange head and rump in females, plump and chunky overall",
    "Pine grosbeak is a bird with wings that are dark blackish-gray with two clean bold white wing bars",
    "Pine grosbeak is a bird with tail that is medium, dark blackish-gray, slightly notched",
    "Pine grosbeak is a bird with legs that are dark, sturdy",
    "A photo of a pine grosbeak, a chunky northern finch where males are rosy-red overall and females are gray with yellowish-orange heads, both showing white wing bars",
]
CUB_CLAUDE["pine_warbler"] = [
    "Pine warbler is a bird with beak that is medium, slender, slightly thicker than other warblers, dark with paler base",
    "Pine warbler is a bird with head that is olive-yellow in males with a faint yellow eyebrow and dark eye-line",
    "Pine warbler is a bird with body that is olive-yellow above, bright lemon-yellow throat and chest in males blending to white belly with faint dark side streaks",
    "Pine warbler is a bird with wings that are dark olive-gray with two clean white wing bars",
    "Pine warbler is a bird with tail that is medium, dark olive-gray with white spots on outer feathers",
    "Pine warbler is a bird with legs that are dark, fairly stout",
    "A photo of a pine warbler, a robust olive-and-yellow warbler with white belly, two clean white wing bars, and faint dark side streaks",
]
CUB_CLAUDE["pomarine_jaeger"] = [
    "Pomarine jaeger is a bird with beak that is medium, hooked, dark gray-black, fairly thick",
    "Pomarine jaeger is a bird with head that is dark blackish cap contrasting with white face and yellow-tinged neck (light morph), all-dark in dark morph",
    "Pomarine jaeger is a bird with body that is white below in light morph adults with a dark breast band, dark brown overall in dark morph",
    "Pomarine jaeger is a bird with wings that are dark brown, broad, with white flash at base of primaries from below",
    "Pomarine jaeger is a bird with tail that is distinctive - elongated central feathers ending in twisted spoon-shaped tips in adults",
    "Pomarine jaeger is a bird with legs that are short, dark blackish",
    "A photo of a pomarine jaeger, a stocky predatory seabird with broad pointed wings and twisted spoon-shaped central tail feathers; light morph shows white belly with dark breast band",
]
CUB_CLAUDE["prairie_warbler"] = [
    "Prairie warbler is a bird with beak that is small, slender, sharply pointed, dark gray-black",
    "Prairie warbler is a bird with head that is yellow face with thin black streaks below the eye and a distinctive black crescent under the eye",
    "Prairie warbler is a bird with body that is olive-yellow above with chestnut streaks on the back, brilliant yellow below with bold black side streaks",
    "Prairie warbler is a bird with wings that are dark with two faint pale wing bars",
    "Prairie warbler is a bird with tail that is medium, often pumped, dark with white spots on outer tail feathers",
    "Prairie warbler is a bird with legs that are dark, slender",
    "A photo of a prairie warbler, a small bright-yellow warbler with chestnut back streaks, bold black side streaks, black face crescent, and a constantly bobbing tail",
]
CUB_CLAUDE["prothonotary_warbler"] = [
    "Prothonotary warbler is a bird with beak that is fairly large, slender, sharply pointed, all black",
    "Prothonotary warbler is a bird with head that is brilliant glowing golden-yellow head and chest, with sharp dark eye-ring",
    "Prothonotary warbler is a bird with body that is brilliant glowing golden-yellow head and underparts, blue-gray back, white undertail coverts",
    "Prothonotary warbler is a bird with wings that are blue-gray without wing bars",
    "Prothonotary warbler is a bird with tail that is short, blue-gray with conspicuous large white outer tail feathers",
    "Prothonotary warbler is a bird with legs that are blackish, sturdy",
    "A photo of a prothonotary warbler, a striking warbler with brilliant glowing golden-yellow head and underparts, blue-gray back and wings, and a black bill",
]
CUB_CLAUDE["purple_finch"] = [
    "Purple finch is a bird with beak that is conical, sharp, dark gray with paler base",
    "Purple finch is a bird with head that is rich raspberry-red wash over crown, face and throat in males (raspberry-dipped look), brown with white eyebrow in females",
    "Purple finch is a bird with body that is raspberry-red overall in males with brown back streaks, brown-and-white-streaked in females",
    "Purple finch is a bird with wings that are brown with two pinkish-buff wing bars in males, brown with pale wing bars in females",
    "Purple finch is a bird with tail that is short, slightly notched, brown",
    "Purple finch is a bird with legs that are dark, slender",
    "A photo of a purple finch, a chunky finch where males are washed in raspberry-red over head and chest (looks dipped in raspberry juice); females are heavily streaked brown-and-white",
]


# ============== R ==============
CUB_CLAUDE["red_bellied_woodpecker"] = [
    "Red bellied woodpecker is a bird with beak that is long, straight, chisel-tipped, dark gray-black",
    "Red bellied woodpecker is a bird with head that is full red crown extending down nape in males, red only on the nape in females, white face",
    "Red bellied woodpecker is a bird with body that is finely barred black-and-white back ('zebra back'), pale gray-buff face and belly with a faint reddish wash on the lower belly",
    "Red bellied woodpecker is a bird with wings that are barred black-and-white with a small white wing patch",
    "Red bellied woodpecker is a bird with tail that is medium, black with white-barred outer feathers",
    "Red bellied woodpecker is a bird with legs that are gray, with two toes forward and two back",
    "A photo of a red bellied woodpecker, a medium woodpecker with a finely zebra-barred black-and-white back, pale belly, and a red crown extending to the nape (full red in males, partial in females)",
]
CUB_CLAUDE["red_breasted_merganser"] = [
    "Red breasted merganser is a bird with beak that is long, slender, hooked at tip with serrated edges (saw-like for catching fish), bright orange-red",
    "Red breasted merganser is a bird with head that is glossy iridescent dark green with a shaggy double-pointed crest in males, warm rusty-cinnamon with shaggy crest in females",
    "Red breasted merganser is a bird with body that is dark above with a chestnut breast and white belly in males, gray-and-rust mix in females",
    "Red breasted merganser is a bird with wings that are dark with a large white wing patch",
    "Red breasted merganser is a bird with tail that is medium, dark gray",
    "Red breasted merganser is a bird with legs that are bright orange-red, with fully webbed feet for diving",
    "A photo of a red breasted merganser, a slim diving duck with a long thin red bill and shaggy double crest; males show dark green head and chestnut breast, females rusty-headed",
]
CUB_CLAUDE["red_cockaded_woodpecker"] = [
    "Red cockaded woodpecker is a bird with beak that is medium, straight, chisel-tipped, dark gray-black",
    "Red cockaded woodpecker is a bird with head that is striking with a black cap, large white face panel (cheek), and a tiny red 'cockade' patch above the cheek in males (often hidden)",
    "Red cockaded woodpecker is a bird with body that is finely barred black-and-white above (zebra back), white below with black-spotted flanks",
    "Red cockaded woodpecker is a bird with wings that are black with white spots arranged in rows",
    "Red cockaded woodpecker is a bird with tail that is stiff, black with white outer feathers",
    "Red cockaded woodpecker is a bird with legs that are gray, with two toes forward and two back",
    "A photo of a red cockaded woodpecker, an endangered black-and-white woodpecker with a striking large white cheek panel and a tiny red cockade patch in males (often hidden)",
]
CUB_CLAUDE["red_eyed_vireo"] = [
    "Red eyed vireo is a bird with beak that is medium, thick, slightly hooked, dark gray with paler base",
    "Red eyed vireo is a bird with head that is olive-green crown bordered by a thin black line, with a bright white eyebrow and red eye in adults",
    "Red eyed vireo is a bird with body that is olive-green above, white below without yellow tones",
    "Red eyed vireo is a bird with wings that are plain olive without obvious wing bars (key distinction from other vireos)",
    "Red eyed vireo is a bird with tail that is short, plain olive",
    "Red eyed vireo is a bird with legs that are blue-gray, sturdy",
    "A photo of a red eyed vireo, a slim olive-green vireo with white eyebrow bordered by black, ruby-red eye in adults, and no wing bars",
]
CUB_CLAUDE["red_faced_cormorant"] = [
    "Red faced cormorant is a bird with beak that is slender, hooked, mostly horn-yellow",
    "Red faced cormorant is a bird with head that is glossy dark blue-green with a striking large bare red patch covering the face and base of bill",
    "Red faced cormorant is a bird with body that is uniformly slim glossy dark iridescent blue-green-black overall, with a white flank patch in breeding plumage",
    "Red faced cormorant is a bird with wings that are dark, fairly narrow",
    "Red faced cormorant is a bird with tail that is medium, stiff, dark",
    "Red faced cormorant is a bird with legs that are dark, with webbed feet",
    "A photo of a red faced cormorant, a slim glossy dark cormorant with a striking large bare red face patch and a yellow bill",
]
CUB_CLAUDE["red_headed_woodpecker"] = [
    "Red headed woodpecker is a bird with beak that is long, straight, chisel-tipped, pale gray turning blackish toward the tip",
    "Red headed woodpecker is a bird with head that is entirely brilliant crimson-red covering the entire head and upper neck, sharply demarcated from white below",
    "Red headed woodpecker is a bird with body that is glossy blue-black upperparts, completely white underparts and rump, sharp tricolor pattern overall",
    "Red headed woodpecker is a bird with wings that are blue-black with large white patches on the secondaries, creating a striking white square in flight",
    "Red headed woodpecker is a bird with tail that is medium, stiff, blue-black, used as a prop against tree trunks while climbing",
    "Red headed woodpecker is a bird with legs that are short, gray, with two toes forward and two backward (zygodactyl) for vertical climbing",
    "A photo of a red headed woodpecker, a striking red-white-and-black woodpecker with a fully crimson head",
]
CUB_CLAUDE["red_legged_kittiwake"] = [
    "Red legged kittiwake is a bird with beak that is medium, all yellow",
    "Red legged kittiwake is a bird with head that is white with a dark eye, occasional dark smudge behind eye in winter",
    "Red legged kittiwake is a bird with body that is white below, darker pearl-gray mantle than Black-legged Kittiwake, slightly stockier build",
    "Red legged kittiwake is a bird with wings that are dark gray with black wingtips",
    "Red legged kittiwake is a bird with tail that is white, square-tipped",
    "Red legged kittiwake is a bird with legs that are bright coral-red (the namesake feature, vs black on Black-legged Kittiwake)",
    "A photo of a red legged kittiwake, a Bering Sea gull with darker gray mantle, black wingtips, white head and underparts, all-yellow bill, and bright coral-red legs",
]
CUB_CLAUDE["red_winged_blackbird"] = [
    "Red winged blackbird is a bird with beak that is conical, sharply pointed, all black",
    "Red winged blackbird is a bird with head that is glossy black overall in males, brown-and-buff streaked with pale eyebrow in females (looks like large sparrow)",
    "Red winged blackbird is a bird with body that is glossy jet-black overall in males, heavily streaked brown-and-white in females",
    "Red winged blackbird is a bird with wings that are glossy black in males with a striking bright red 'epaulet' shoulder patch bordered by yellow; brown-streaked in females",
    "Red winged blackbird is a bird with tail that is medium, glossy black in males, brown in females",
    "Red winged blackbird is a bird with legs that are dark, sturdy, fairly long",
    "A photo of a red winged blackbird, a familiar marsh blackbird where males are glossy black with a bright red-and-yellow shoulder epaulet; females are sparrow-like brown-and-white-streaked",
]
CUB_CLAUDE["rhinoceros_auklet"] = [
    "Rhinoceros auklet is a bird with beak that is large, conical, orange-yellow with a distinctive small upturned 'horn' projecting from the upper base in breeding plumage",
    "Rhinoceros auklet is a bird with head that is dark gray-brown with two thin white plumes (one above eye, one below) in breeding plumage",
    "Rhinoceros auklet is a bird with body that is dark gray-brown above, paler gray-brown below",
    "Rhinoceros auklet is a bird with wings that are short, narrow, dark, used both for flying and underwater swimming",
    "Rhinoceros auklet is a bird with tail that is short, dark",
    "Rhinoceros auklet is a bird with legs that are short, blackish, with webbed feet",
    "A photo of a rhinoceros auklet, a stocky dark seabird with a yellow bill bearing a distinctive upturned 'horn' projection and two thin white facial plumes in breeding plumage",
]
CUB_CLAUDE["ring_billed_gull"] = [
    "Ring billed gull is a bird with beak that is medium length, yellow with a distinct complete black ring near the tip",
    "Ring billed gull is a bird with head that is white in breeding adults, finely streaked dusky in winter, with a pale yellow eye",
    "Ring billed gull is a bird with body that is white below, pale gray mantle (paler than Herring), juveniles mottled brown",
    "Ring billed gull is a bird with wings that are pale gray with black wingtips marked with white",
    "Ring billed gull is a bird with tail that is white in adults, dark-banded in immatures",
    "Ring billed gull is a bird with legs that are yellow, sturdy",
    "A photo of a ring billed gull, a medium pale-gray gull with yellow legs, pale yellow eyes, and a yellow bill marked by a distinct complete black ring",
]
CUB_CLAUDE["ringed_kingfisher"] = [
    "Ringed kingfisher is a bird with beak that is huge, long, dagger-like, all dark gray-black",
    "Ringed kingfisher is a bird with head that is large with a shaggy blue-gray crest, white throat",
    "Ringed kingfisher is a bird with body that is slate-blue-gray above, with a brilliant rufous-cinnamon belly and chest in both sexes (only females show a blue chest band)",
    "Ringed kingfisher is a bird with wings that are blue-gray, broad",
    "Ringed kingfisher is a bird with tail that is medium, blue-gray with white bars",
    "Ringed kingfisher is a bird with legs that are very short, dark",
    "A photo of a ringed kingfisher, a large blue-gray kingfisher with a shaggy crest, dagger bill, and brilliant rufous-cinnamon underparts (largest American kingfisher)",
]
CUB_CLAUDE["rock_wren"] = [
    "Rock wren is a bird with beak that is long, slender, slightly downcurved, dark with paler base",
    "Rock wren is a bird with head that is plain pale gray-brown with a faint pale eyebrow, dark eye stripe",
    "Rock wren is a bird with body that is grayish-brown above with fine pale streaks, pale buff below with very fine breast streaking",
    "Rock wren is a bird with wings that are gray-brown with very fine darker barring",
    "Rock wren is a bird with tail that is medium, often cocked or bobbed, with rusty-buff outer corners and a dark band near tip",
    "Rock wren is a bird with legs that are pinkish-buff, fairly long, suited for hopping on rocky terrain",
    "A photo of a rock wren, a pale gray-brown wren of rocky habitats with a long downcurved bill, faint streaking, and rusty-buff tail corners",
]
CUB_CLAUDE["rose_breasted_grosbeak"] = [
    "Rose breasted grosbeak is a bird with beak that is large, thick, conical, ivory-white",
    "Rose breasted grosbeak is a bird with head that is solid jet-black in males, brown-and-white striped (like a large sparrow) in females",
    "Rose breasted grosbeak is a bird with body that is jet-black above with white below in males with a striking bright rose-red triangular breast patch; brown-streaked in females",
    "Rose breasted grosbeak is a bird with wings that are black with bold white wing patches and bars in males, brown with two buff wing bars in females",
    "Rose breasted grosbeak is a bird with tail that is medium, black with white outer corners in males",
    "Rose breasted grosbeak is a bird with legs that are dark blue-gray, sturdy",
    "A photo of a rose breasted grosbeak, a striking large grosbeak where males are black-and-white with a brilliant rose-red triangular breast patch; females are heavily streaked brown-and-white",
]
CUB_CLAUDE["ruby_throated_hummingbird"] = [
    "Ruby throated hummingbird is a bird with beak that is very long, needle-thin, straight, entirely black, adapted for flower nectar feeding",
    "Ruby throated hummingbird is a bird with head that is metallic emerald-green crown, with adult males showing an iridescent ruby-red gorget under the throat",
    "Ruby throated hummingbird is a bird with body that is bright metallic green upperparts, grayish-white underparts, females and juveniles lacking the red throat",
    "Ruby throated hummingbird is a bird with wings that are dark, very narrow, pointed, beating extremely fast and producing the characteristic humming sound",
    "Ruby throated hummingbird is a bird with tail that is short, forked in males, rounded with white tips in females, dusky blackish overall",
    "Ruby throated hummingbird is a bird with legs that are tiny, almost invisible in flight, dark, used only for perching not walking",
    "A photo of a ruby throated hummingbird, a tiny iridescent green hummer where adult males show a flashing red throat patch",
]
CUB_CLAUDE["rufous_hummingbird"] = [
    "Rufous hummingbird is a bird with beak that is medium for a hummingbird, straight, slender, all black",
    "Rufous hummingbird is a bird with head that is bright orange-rufous in males with iridescent flame-red gorget, green crown in females",
    "Rufous hummingbird is a bird with body that is uniformly bright copper-rufous above and below in adult males (very orange overall), green-and-rufous in females",
    "Rufous hummingbird is a bird with wings that are dark, narrow, pointed, beating extremely fast",
    "Rufous hummingbird is a bird with tail that is rufous in males, green-rufous in females",
    "Rufous hummingbird is a bird with legs that are tiny, dark, used only for perching",
    "A photo of a rufous hummingbird, a small fiery copper-rufous hummingbird where adult males are brilliant orange overall with a flame-red gorget",
]
CUB_CLAUDE["rusty_blackbird"] = [
    "Rusty blackbird is a bird with beak that is conical, sharply pointed, all black",
    "Rusty blackbird is a bird with head that is glossy black with pale yellow eye in breeding males, gray-and-rust 'rusty' look with pale eye in winter (the namesake)",
    "Rusty blackbird is a bird with body that is glossy black overall in breeding males; both sexes show extensive rust-tipped feathers giving a 'rusty' appearance in fall and winter",
    "Rusty blackbird is a bird with wings that are glossy black in breeding, rust-tipped feathers in winter",
    "Rusty blackbird is a bird with tail that is medium, glossy black",
    "Rusty blackbird is a bird with legs that are dark, sturdy, fairly long",
    "A photo of a rusty blackbird, a slim blackbird with pale yellow eyes - glossy black in breeding males but heavily rust-tipped giving a rusty appearance in fall and winter plumage",
]

# ============== S ==============
CUB_CLAUDE["sage_thrasher"] = [
    "Sage thrasher is a bird with beak that is shorter than other thrashers, slender, slightly downcurved, dark gray",
    "Sage thrasher is a bird with head that is plain gray-brown with thin pale eyebrow, distinctive bright yellow eye",
    "Sage thrasher is a bird with body that is plain gray-brown above, white below with bold dark streaks (not spots) on the breast and flanks",
    "Sage thrasher is a bird with wings that are gray-brown with thin pale wing bars",
    "Sage thrasher is a bird with tail that is medium, gray-brown with white tips on outer feathers",
    "Sage thrasher is a bird with legs that are dark, fairly long",
    "A photo of a sage thrasher, the smallest thrasher, a sagebrush-country bird with plain gray-brown upperparts, bold dark breast streaks, and bright yellow eyes",
]
CUB_CLAUDE["savannah_sparrow"] = [
    "Savannah sparrow is a bird with beak that is small, conical, sharp, pinkish-horn",
    "Savannah sparrow is a bird with head that is finely streaked brown crown with a distinctive yellow patch over and in front of the eye, white central crown stripe",
    "Savannah sparrow is a bird with body that is heavily streaked brown above and white below with bold thin dark streaks across the breast",
    "Savannah sparrow is a bird with wings that are brown with rusty edges, no obvious wing bars",
    "Savannah sparrow is a bird with tail that is short, slightly notched, brown",
    "Savannah sparrow is a bird with legs that are pinkish, fairly short",
    "A photo of a savannah sparrow, a streaked brown grassland sparrow with bold breast streaks, a yellow eyebrow patch in front of the eye, and a notched tail",
]
CUB_CLAUDE["sayornis"] = [
    "Sayornis is a bird with beak that is broad-based, all black, slightly wider than other small flycatchers",
    "Sayornis is a bird with head that is plain dark grayish-brown without obvious markings (Eastern Phoebe) or warm gray-brown crown (Say's Phoebe)",
    "Sayornis is a bird with body that is grayish-brown above, dingy white below in Eastern; warm gray-brown above with rich cinnamon-buff belly in Say's",
    "Sayornis is a bird with wings that are dark olive-brown without obvious wing bars",
    "Sayornis is a bird with tail that is medium, dark, often pumped up and down constantly (key behavior of phoebes)",
    "Sayornis is a bird with legs that are dark, slender",
    "A photo of a sayornis (phoebe), a plain medium-sized flycatcher that constantly pumps its tail down then up; species can be plain (Eastern) or warm cinnamon-bellied (Say's)",
]
CUB_CLAUDE["scarlet_tanager"] = [
    "Scarlet tanager is a bird with beak that is medium, thick, slightly notched, ivory-pale gray",
    "Scarlet tanager is a bird with head that is brilliant scarlet-red in breeding males, plain olive-yellow in females and non-breeding males",
    "Scarlet tanager is a bird with body that is brilliant flame-scarlet body with sharply contrasting jet-black wings and tail in males; uniformly olive-yellow in females",
    "Scarlet tanager is a bird with wings that are jet-black in breeding males contrasting with red body, olive-brown in females",
    "Scarlet tanager is a bird with tail that is medium, jet-black in breeding males, olive-brown in females",
    "Scarlet tanager is a bird with legs that are gray, sturdy",
    "A photo of a scarlet tanager, a striking songbird where breeding males show brilliant flame-red body sharply contrasting with jet-black wings and tail; females are uniformly olive-yellow",
]
CUB_CLAUDE["scissor_tailed_flycatcher"] = [
    "Scissor tailed flycatcher is a bird with beak that is medium, broad-based, all black",
    "Scissor tailed flycatcher is a bird with head that is pearly pale gray with darker eye, no obvious face pattern",
    "Scissor tailed flycatcher is a bird with body that is pearly gray above, white below with salmon-pink wash on the flanks and belly",
    "Scissor tailed flycatcher is a bird with wings that are dark gray with salmon-pink wing-linings flashing in flight",
    "Scissor tailed flycatcher is a bird with tail that is extraordinarily long, deeply forked like an open scissor, gray with white outer feathers",
    "Scissor tailed flycatcher is a bird with legs that are dark, slender, short",
    "A photo of a scissor tailed flycatcher, an unmistakable pale-gray flycatcher with extraordinarily long, deeply forked scissor-like tail and salmon-pink flanks",
]
CUB_CLAUDE["scott_oriole"] = [
    "Scott oriole is a bird with beak that is long, slender, slightly downcurved, dark gray-black",
    "Scott oriole is a bird with head that is solid jet-black hood (head, throat, upper back) in adult males, plain olive-gray in females",
    "Scott oriole is a bird with body that is brilliant lemon-yellow body with black hood and wings in males, dull olive-yellow with grayer back in females",
    "Scott oriole is a bird with wings that are black with a single white wing bar in males, dull olive in females",
    "Scott oriole is a bird with tail that is long, mostly black with yellow base in males",
    "Scott oriole is a bird with legs that are blue-gray, sturdy",
    "A photo of a scott oriole, a slim oriole of arid southwest where males show solid jet-black hood and back contrasting with brilliant lemon-yellow body",
]
CUB_CLAUDE["seaside_sparrow"] = [
    "Seaside sparrow is a bird with beak that is large for a sparrow, conical, dark gray with paler base",
    "Seaside sparrow is a bird with head that is dull gray-brown with a striking bright yellow patch in front of the eye, gray ear patch, and white throat with dark moustache stripe",
    "Seaside sparrow is a bird with body that is uniformly dark gray-brown above without bright markings, gray-buff below with faint streaks",
    "Seaside sparrow is a bird with wings that are dark gray-brown with no obvious markings",
    "Seaside sparrow is a bird with tail that is short, with stiff pointed feathers, brown",
    "Seaside sparrow is a bird with legs that are pinkish-flesh, sturdy",
    "A photo of a seaside sparrow, a dark gray-brown salt marsh sparrow with a striking bright yellow patch in front of the eye and a sharply demarcated white throat",
]
CUB_CLAUDE["shiny_cowbird"] = [
    "Shiny cowbird is a bird with beak that is conical, sharply pointed, slightly slimmer than other cowbirds, all black",
    "Shiny cowbird is a bird with head that is glossy iridescent purplish-blue in males with a dark eye, plain dull gray-brown in females",
    "Shiny cowbird is a bird with body that is uniformly glossy iridescent purplish-blue all over in males (lacking brown head of Brown-headed Cowbird), plain dull gray-brown in females",
    "Shiny cowbird is a bird with wings that are glossy purplish-blue in males, gray-brown in females",
    "Shiny cowbird is a bird with tail that is medium, glossy purplish-blue in males, gray-brown in females",
    "Shiny cowbird is a bird with legs that are dark, sturdy",
    "A photo of a shiny cowbird, a slim cowbird where males are uniformly glossy purplish-blue all over (lacking the brown head of Brown-headed Cowbird), females are plain gray-brown",
]
CUB_CLAUDE["slaty_backed_gull"] = [
    "Slaty backed gull is a bird with beak that is large, yellow with red spot near tip on lower mandible",
    "Slaty backed gull is a bird with head that is white in breeding, dusky-streaked in winter, with dark eye and pink orbital ring",
    "Slaty backed gull is a bird with body that is white below, dark slate-blackish mantle (very dark for a gull)",
    "Slaty backed gull is a bird with wings that are very dark slate-black with a striking 'string of pearls' pattern of white spots on the trailing edge of primaries",
    "Slaty backed gull is a bird with tail that is white in adults, dark-banded in immatures",
    "Slaty backed gull is a bird with legs that are bright pink, sturdy",
    "A photo of a slaty backed gull, a large dark-mantled gull with very dark slate-black back, distinctive 'string of pearls' white spots on the wing trailing edge, pink legs",
]
CUB_CLAUDE["song_sparrow"] = [
    "Song sparrow is a bird with beak that is conical, sharp, dark with paler base",
    "Song sparrow is a bird with head that is brown crown with rusty stripes and pale gray central stripe, distinct dark moustache stripe, gray cheek",
    "Song sparrow is a bird with body that is rusty-brown above with bold dark streaks, white below with heavy bold dark streaks converging into a central breast spot",
    "Song sparrow is a bird with wings that are rusty-brown without obvious wing bars",
    "Song sparrow is a bird with tail that is medium-long, brown, often pumped",
    "Song sparrow is a bird with legs that are pinkish-buff, fairly long",
    "A photo of a song sparrow, a streaky-brown sparrow with bold rusty-and-brown streaking on the breast and flanks converging into a central dark spot",
]
CUB_CLAUDE["sooty_albatross"] = [
    "Sooty albatross is a bird with beak that is long, hooked, all black with a yellow groove (sulcus) along the lower mandible",
    "Sooty albatross is a bird with head that is uniformly dark sooty-brown with a thin white eye-ring",
    "Sooty albatross is a bird with body that is uniformly slim and elegant dark chocolate-brown overall, no contrasting markings",
    "Sooty albatross is a bird with wings that are very long, narrow, pointed, all dark brown, built for dynamic soaring",
    "Sooty albatross is a bird with tail that is long, wedge-shaped (pointed), dark brown",
    "Sooty albatross is a bird with legs that are short, blackish, with webbed feet for swimming",
    "A photo of a sooty albatross, a slim elegant all-dark-brown albatross with very long pointed wings, white eye-ring, yellow bill stripe, and a wedge-shaped tail",
]
CUB_CLAUDE["spotted_catbird"] = [
    "Spotted catbird is a bird with beak that is short, slightly hooked, dark gray-black",
    "Spotted catbird is a bird with head that is olive-green with bold pale spots overall and rusty highlights, sparse dark face markings",
    "Spotted catbird is a bird with body that is olive-green above with white spots, white below with bold black spots and bars",
    "Spotted catbird is a bird with wings that are olive-green with white spotting on coverts and flight feathers",
    "Spotted catbird is a bird with tail that is medium, olive-green with white tips on outer feathers",
    "Spotted catbird is a bird with legs that are dark, sturdy",
    "A photo of a spotted catbird, a stocky olive-green Australian rainforest bird heavily marked with bold white spots above and bold black spots below",
]
CUB_CLAUDE["summer_tanager"] = [
    "Summer tanager is a bird with beak that is medium, thick, ivory-yellowish",
    "Summer tanager is a bird with head that is uniformly rich rosy-red in adult males, plain warm yellow-olive in females",
    "Summer tanager is a bird with body that is uniformly rich rosy-red overall in males without contrasting wings (key distinction from Scarlet Tanager); plain mustard-yellow overall in females",
    "Summer tanager is a bird with wings that are rosy-red in males (matching the body), olive-yellow in females",
    "Summer tanager is a bird with tail that is medium, rosy-red in males, yellow-olive in females",
    "Summer tanager is a bird with legs that are gray, sturdy",
    "A photo of a summer tanager, a medium tanager where adult males are uniformly rosy-red all over (including wings, unlike Scarlet Tanager) and females are uniformly mustard-yellow",
]
CUB_CLAUDE["swainson_warbler"] = [
    "Swainson warbler is a bird with beak that is fairly long, slender, slightly downcurved, dark gray with paler base",
    "Swainson warbler is a bird with head that is rich brown crown with a long pale buff eyebrow stripe and a thin dark eye-line",
    "Swainson warbler is a bird with body that is plain warm brown above, dingy buffy-white below without streaks (very plain)",
    "Swainson warbler is a bird with wings that are plain brown without obvious wing bars",
    "Swainson warbler is a bird with tail that is short, plain brown",
    "Swainson warbler is a bird with legs that are pinkish-flesh, fairly long",
    "A photo of a swainson warbler, a secretive plain warm-brown warbler with a long buff eyebrow, plain unstreaked underparts, and a long bill for a warbler",
]

# ============== T ==============
CUB_CLAUDE["tennessee_warbler"] = [
    "Tennessee warbler is a bird with beak that is small, sharply pointed, slightly thicker than other warblers, dark with paler base",
    "Tennessee warbler is a bird with head that is gray crown contrasting with olive-green back in breeding males, with a fine pale eyebrow",
    "Tennessee warbler is a bird with body that is olive-green above, white below in breeding males (yellower in females and fall birds), without streaks",
    "Tennessee warbler is a bird with wings that are olive without obvious wing bars in breeding plumage",
    "Tennessee warbler is a bird with tail that is short, olive, often appears short and stubby",
    "Tennessee warbler is a bird with legs that are dark, slender",
    "A photo of a tennessee warbler, a small plain olive-green warbler with a gray hood, fine pale eyebrow, and clean white belly with no streaks or wing bars",
]
CUB_CLAUDE["tree_sparrow"] = [
    "Tree sparrow is a bird with beak that is small, conical, sharp, dark above and yellow below (bicolored)",
    "Tree sparrow is a bird with head that is solid rusty-rufous crown, gray face with a small dark central spot on the white-gray breast",
    "Tree sparrow is a bird with body that is gray above with rusty-streaked back, plain pale gray below with a single distinctive dark central breast spot",
    "Tree sparrow is a bird with wings that are brown with two crisp white wing bars",
    "Tree sparrow is a bird with tail that is medium, slightly notched, brown",
    "Tree sparrow is a bird with legs that are pinkish-buff, fairly short",
    "A photo of a tree sparrow, a small wintering sparrow with a solid rusty crown, bicolored bill (dark above, yellow below), and a single dark central breast spot",
]
CUB_CLAUDE["tree_swallow"] = [
    "Tree swallow is a bird with beak that is very short, broad-based, blackish",
    "Tree swallow is a bird with head that is glossy iridescent metallic blue-green crown extending down to just below the eye",
    "Tree swallow is a bird with body that is glossy iridescent blue-green above (electric blue-green in good light), entirely clean white below",
    "Tree swallow is a bird with wings that are dark with iridescent edges, long, pointed, built for graceful flight",
    "Tree swallow is a bird with tail that is short, slightly notched, dark",
    "Tree swallow is a bird with legs that are very short, dark, weak",
    "A photo of a tree swallow, a small streamlined swallow with brilliant iridescent blue-green upperparts and clean white underparts",
]
CUB_CLAUDE["tropical_kingbird"] = [
    "Tropical kingbird is a bird with beak that is large, broad, all dark, slightly hooked",
    "Tropical kingbird is a bird with head that is plain gray with a darker gray ear patch",
    "Tropical kingbird is a bird with body that is gray above, pale gray throat, brilliant lemon-yellow belly with olive wash on the chest",
    "Tropical kingbird is a bird with wings that are dark gray-brown",
    "Tropical kingbird is a bird with tail that is medium, slightly notched, brown",
    "Tropical kingbird is a bird with legs that are dark, slender",
    "A photo of a tropical kingbird, a yellow-bellied gray kingbird similar to Western Kingbird but with a notched tail (no white outer tail feathers), heavy bill, and yellower belly",
]

# ============== V ==============
CUB_CLAUDE["vermilion_flycatcher"] = [
    "Vermilion flycatcher is a bird with beak that is small, broad-based, all black",
    "Vermilion flycatcher is a bird with head that is brilliant flame-red in adult males with a dark blackish-brown mask through eye, dull gray-brown in females",
    "Vermilion flycatcher is a bird with body that is brilliant flame-red crown and underparts in males, blackish-brown back; gray-brown above and pale buffy below with peach-pink wash in females",
    "Vermilion flycatcher is a bird with wings that are blackish-brown in males, gray-brown in females",
    "Vermilion flycatcher is a bird with tail that is medium, often spread when displaying, blackish in males",
    "Vermilion flycatcher is a bird with legs that are blackish, slender",
    "A photo of a vermilion flycatcher, a small flycatcher where adult males are brilliant flame-red with blackish mask and back; females are plain gray-brown with peach-pink wash",
]
CUB_CLAUDE["vesper_sparrow"] = [
    "Vesper sparrow is a bird with beak that is conical, sharp, pinkish-horn",
    "Vesper sparrow is a bird with head that is brown-streaked crown with a thin white eye-ring (giving 'wide-eyed' look) and a distinctive bright chestnut shoulder patch (often hidden)",
    "Vesper sparrow is a bird with body that is grayish-brown with crisp dark streaks above, white below with brown breast streaks",
    "Vesper sparrow is a bird with wings that are brown with a small chestnut shoulder patch (lesser coverts), often visible only when bird is alert",
    "Vesper sparrow is a bird with tail that is medium, dark with conspicuous bright white outer tail feathers flashing in flight (key field mark)",
    "Vesper sparrow is a bird with legs that are pinkish-buff, fairly long",
    "A photo of a vesper sparrow, a streaky-brown grassland sparrow with a complete white eye-ring, chestnut shoulder patch (often hidden), and conspicuous white-flashing outer tail feathers",
]


# ============== W ==============
CUB_CLAUDE["warbling_vireo"] = [
    "Warbling vireo is a bird with beak that is short, thick, slightly hooked, dark gray with paler base",
    "Warbling vireo is a bird with head that is plain pale gray with a faint pale eyebrow stripe (no black borders unlike Red-eyed Vireo)",
    "Warbling vireo is a bird with body that is plain olive-gray above, pale yellowish-white below (cleaner and paler than other vireos)",
    "Warbling vireo is a bird with wings that are plain olive-gray without obvious wing bars",
    "Warbling vireo is a bird with tail that is short, plain olive-gray",
    "Warbling vireo is a bird with legs that are blue-gray, sturdy",
    "A photo of a warbling vireo, a plain pale olive-gray vireo with a faint pale eyebrow, no wing bars, and clean pale yellowish underparts (the plainest vireo)",
]
CUB_CLAUDE["western_grebe"] = [
    "Western grebe is a bird with beak that is long, slender, sharply pointed, bright greenish-yellow",
    "Western grebe is a bird with head that is striking with a sharply demarcated black crown extending below the red eye, white throat",
    "Western grebe is a bird with body that is glossy black above, snow-white below, with a long slender swan-like neck",
    "Western grebe is a bird with wings that are dark, fairly long, mostly hidden when swimming",
    "Western grebe is a bird with tail that is barely visible, almost vestigial",
    "Western grebe is a bird with legs that are dark, lobed feet set far back, suited for diving",
    "A photo of a western grebe, an elegant slim long-necked grebe with sharply demarcated black crown above red eye, snow-white throat and underparts, and yellow bill",
]
CUB_CLAUDE["western_gull"] = [
    "Western gull is a bird with beak that is large, thick, yellow with a bright red spot on the lower mandible",
    "Western gull is a bird with head that is white in breeding adults, lightly streaked dusky in winter, with a dark eye and small yellow orbital ring",
    "Western gull is a bird with body that is white below, dark slate-gray mantle (darker than Herring Gull)",
    "Western gull is a bird with wings that are dark slate-gray with black wingtips marked with white",
    "Western gull is a bird with tail that is white in adults, dark-banded in immatures",
    "Western gull is a bird with legs that are bright pink, sturdy",
    "A photo of a western gull, a large dark-mantled Pacific gull with very dark slate-gray back, black wingtips, white head and underparts, pink legs, and yellow bill with red spot",
]
CUB_CLAUDE["western_meadowlark"] = [
    "Western meadowlark is a bird with beak that is medium, slender, sharply pointed, gray with darker tip",
    "Western meadowlark is a bird with head that is striking with brown-and-white striped crown and bright yellow lower face/throat (yellow extends into the moustache area unlike Eastern)",
    "Western meadowlark is a bird with body that is mottled brown above with bold streaks, brilliant lemon-yellow below with a striking large black 'V' or 'bib' across the chest",
    "Western meadowlark is a bird with wings that are mottled brown with dark and pale barring",
    "Western meadowlark is a bird with tail that is short, brown with conspicuous white outer tail feathers visible in flight",
    "Western meadowlark is a bird with legs that are pinkish-buff, fairly long",
    "A photo of a western meadowlark, a stocky brown-streaked grassland songbird with brilliant yellow underparts marked by a striking large black V-bib on the chest",
]
CUB_CLAUDE["western_wood_pewee"] = [
    "Western wood pewee is a bird with beak that is medium, broad-based, dark upper mandible and pale yellow-orange lower mandible",
    "Western wood pewee is a bird with head that is plain dark olive-gray with a slight peaked crown but no obvious eye-ring",
    "Western wood pewee is a bird with body that is plain dark olive-gray above and on chest, pale grayish-buff belly with diffuse 'vest' look",
    "Western wood pewee is a bird with wings that are dark with two pale buffy-white wing bars",
    "Western wood pewee is a bird with tail that is medium, dark olive-gray, often slightly notched",
    "Western wood pewee is a bird with legs that are dark, slender",
    "A photo of a western wood pewee, a plain dark olive-gray flycatcher with a slight peaked crown, no eye-ring, and two pale wing bars - looks 'vested' on the chest",
]
CUB_CLAUDE["whip_poor_will"] = [
    "Whip poor will is a bird with beak that is tiny but with a huge gape lined with bristles",
    "Whip poor will is a bird with head that is large, flat, mottled cryptic gray-and-brown with very large dark eyes for nocturnal hunting",
    "Whip poor will is a bird with body that is intricately patterned mottled gray, brown, black, and rust above and below for bark-like camouflage",
    "Whip poor will is a bird with wings that are long, rounded, mottled brown without bold white patches",
    "Whip poor will is a bird with tail that is long, brown with broad white outer corners (males) or buff corners (females)",
    "Whip poor will is a bird with legs that are very short, weak, mostly hidden, used only for perching lengthwise on branches",
    "A photo of a whip poor will, a small cryptic mottled-brown nightjar with massive dark eyes, tiny bill, no white wing patches, and broad white tail corners in males",
]
CUB_CLAUDE["white_breasted_kingfisher"] = [
    "White breasted kingfisher is a bird with beak that is huge, dagger-like, all bright coral-red",
    "White breasted kingfisher is a bird with head that is rich chocolate-brown crown, white throat and breast contrasting sharply",
    "White breasted kingfisher is a bird with body that is brilliant turquoise-blue back and wings, white throat and chest, chestnut belly and flanks",
    "White breasted kingfisher is a bird with wings that are vivid turquoise-blue with conspicuous white wing patches visible in flight",
    "White breasted kingfisher is a bird with tail that is medium, vivid turquoise-blue",
    "White breasted kingfisher is a bird with legs that are short, bright coral-red",
    "A photo of a white breasted kingfisher, a brilliantly colored kingfisher with chocolate head, snow-white breast, vivid turquoise wings and tail, chestnut belly, and a coral-red bill",
]
CUB_CLAUDE["white_breasted_nuthatch"] = [
    "White breasted nuthatch is a bird with beak that is medium, straight, sharply pointed, dark gray-black",
    "White breasted nuthatch is a bird with head that is solid jet-black cap (or gray cap in females) extending down to nape, sharply contrasting with bright white face",
    "White breasted nuthatch is a bird with body that is blue-gray above, clean white below with rusty undertail coverts",
    "White breasted nuthatch is a bird with wings that are blue-gray with darker flight feather edges",
    "White breasted nuthatch is a bird with tail that is short, blue-gray with white spots on outer feathers",
    "White breasted nuthatch is a bird with legs that are dark, very short, with strong claws for climbing trees in any direction (including head-down)",
    "A photo of a white breasted nuthatch, a small compact gray-and-white nuthatch with black cap (gray in females), bright white face, and the habit of climbing down trees head-first",
]
CUB_CLAUDE["white_crowned_sparrow"] = [
    "White crowned sparrow is a bird with beak that is conical, sharp, pinkish-orange to yellowish (varies by subspecies)",
    "White crowned sparrow is a bird with head that is striking with bold black-and-white stripes on the crown - a clean white central stripe, two black side stripes, and a white eyebrow",
    "White crowned sparrow is a bird with body that is gray-brown above with subtle streaks, plain pale gray below without streaks",
    "White crowned sparrow is a bird with wings that are brown with two crisp white wing bars",
    "White crowned sparrow is a bird with tail that is medium, brown",
    "White crowned sparrow is a bird with legs that are pinkish, fairly long",
    "A photo of a white crowned sparrow, a clean elegant sparrow with bold black-and-white stripes on the crown, plain gray underparts, and a pinkish bill",
]
CUB_CLAUDE["white_eyed_vireo"] = [
    "White eyed vireo is a bird with beak that is short, thick, slightly hooked, dark with pale base",
    "White eyed vireo is a bird with head that is olive-green crown with bold yellow spectacles around a piercing white iris (the namesake feature)",
    "White eyed vireo is a bird with body that is olive-green above, white below with bright yellow flanks",
    "White eyed vireo is a bird with wings that are olive with two white wing bars",
    "White eyed vireo is a bird with tail that is short, olive-brown",
    "White eyed vireo is a bird with legs that are blue-gray, sturdy",
    "A photo of a white eyed vireo, a small olive-and-white vireo with bold yellow spectacles around piercing white eyes, two white wing bars, and yellow flanks",
]
CUB_CLAUDE["white_necked_raven"] = [
    "White necked raven is a bird with beak that is huge, thick, slightly hooked, all glossy black with pale tip",
    "White necked raven is a bird with head that is uniformly glossy black",
    "White necked raven is a bird with body that is uniformly glossy black overall with a striking white half-collar across the back of the neck (visible only when ruffled or in flight)",
    "White necked raven is a bird with wings that are very long, broad, glossy black with strongly fingered primaries",
    "White necked raven is a bird with tail that is medium, slightly wedge-shaped, glossy black",
    "White necked raven is a bird with legs that are sturdy, all black",
    "A photo of a white necked raven, a large all-black corvid with a massive bill and a striking concealed white half-collar across the upper back of the neck",
]
CUB_CLAUDE["white_pelican"] = [
    "White pelican is a bird with beak that is enormous, long, with a huge expandable yellow throat pouch, breeding adults grow a horny knob on the upper bill",
    "White pelican is a bird with head that is white with yellow-tinged crown when breeding, large with very long bill",
    "White pelican is a bird with body that is entirely snow-white in adults, very large and stocky",
    "White pelican is a bird with wings that are very long, broad, mostly white with sharply contrasting black flight feathers visible only in flight",
    "White pelican is a bird with tail that is short, white",
    "White pelican is a bird with legs that are bright orange-yellow, with fully webbed feet",
    "A photo of a white pelican, a huge snow-white waterbird with an enormous yellow bill and pouch, orange feet, and contrasting black wing tips visible in flight",
]
CUB_CLAUDE["white_throated_sparrow"] = [
    "White throated sparrow is a bird with beak that is conical, sharp, dark gray",
    "White throated sparrow is a bird with head that is striking with bold black-and-white striped crown (white-striped form) or tan-and-brown striped (tan-striped form), and a brilliant yellow patch in front of the eye",
    "White throated sparrow is a bird with body that is rusty-brown above with streaks, gray below with a sharply demarcated bright white throat patch (the namesake)",
    "White throated sparrow is a bird with wings that are rusty-brown with two thin white wing bars",
    "White throated sparrow is a bird with tail that is medium, brown",
    "White throated sparrow is a bird with legs that are pinkish, fairly long",
    "A photo of a white throated sparrow, a sparrow with bold crown stripes, brilliant yellow eye patch, sharply demarcated bright white throat, and rusty-brown back",
]
CUB_CLAUDE["wilson_warbler"] = [
    "Wilson warbler is a bird with beak that is small, slender, sharply pointed, dark gray-black",
    "Wilson warbler is a bird with head that is brilliant lemon-yellow face crowned by a striking small jet-black skullcap in males (faint or absent in females)",
    "Wilson warbler is a bird with body that is olive-yellow above, brilliant lemon-yellow below",
    "Wilson warbler is a bird with wings that are plain olive without obvious wing bars",
    "Wilson warbler is a bird with tail that is medium, plain olive-yellow, often flicked",
    "Wilson warbler is a bird with legs that are pinkish, slender",
    "A photo of a wilson warbler, a tiny brilliantly yellow warbler where males show a striking small jet-black skullcap on the crown",
]
CUB_CLAUDE["winter_wren"] = [
    "Winter wren is a bird with beak that is short, slender, slightly downcurved, dark",
    "Winter wren is a bird with head that is plain warm dark brown with a faint pale eyebrow, smaller and darker than House Wren",
    "Winter wren is a bird with body that is uniformly dark warm brown above with very fine dark barring, paler buff-brown below with dark barring on flanks and belly",
    "Winter wren is a bird with wings that are warm brown with fine dark barring",
    "Winter wren is a bird with tail that is very short and stubby (much shorter than House Wren), held cocked vertically",
    "Winter wren is a bird with legs that are pinkish-buff, short",
    "A photo of a winter wren, a tiny dark warm-brown wren with a very short stubby cocked tail, fine dark barring on flanks, and faint pale eyebrow",
]
CUB_CLAUDE["worm_eating_warbler"] = [
    "Worm eating warbler is a bird with beak that is medium, slender, sharply pointed, pale pinkish-flesh",
    "Worm eating warbler is a bird with head that is buffy-cream with bold black stripes - black crown stripes on either side of a buffy central stripe, and a black eye-line",
    "Worm eating warbler is a bird with body that is plain olive-buff above, pale warm buff below without streaks",
    "Worm eating warbler is a bird with wings that are plain olive without obvious wing bars",
    "Worm eating warbler is a bird with tail that is short, plain olive",
    "Worm eating warbler is a bird with legs that are pinkish-buff, fairly long",
    "A photo of a worm eating warbler, a plain warm-buff-and-olive warbler with striking bold black-and-buff stripes on the crown",
]

# ============== Y ==============
CUB_CLAUDE["yellow_bellied_flycatcher"] = [
    "Yellow bellied flycatcher is a bird with beak that is small, broad-based, dark upper and pale orange-pink lower mandible",
    "Yellow bellied flycatcher is a bird with head that is olive-green with a complete bold yellow eye-ring (key feature)",
    "Yellow bellied flycatcher is a bird with body that is olive-green above, brilliant lemon-yellow throat, chest, and belly (yellower than other empidonax flycatchers)",
    "Yellow bellied flycatcher is a bird with wings that are dark with two clean yellow-tinged wing bars",
    "Yellow bellied flycatcher is a bird with tail that is short, dark olive",
    "Yellow bellied flycatcher is a bird with legs that are dark, slender",
    "A photo of a yellow bellied flycatcher, a small olive-green empidonax flycatcher with a complete yellow eye-ring and uniquely bright yellow underparts including the throat",
]
CUB_CLAUDE["yellow_billed_cuckoo"] = [
    "Yellow billed cuckoo is a bird with beak that is long, slightly downcurved, with a striking yellow lower mandible (yellow-billed)",
    "Yellow billed cuckoo is a bird with head that is plain warm brown crown without obvious face pattern, dark eye",
    "Yellow billed cuckoo is a bird with body that is plain warm brown above, white below without markings",
    "Yellow billed cuckoo is a bird with wings that are warm brown with rich rusty-rufous primary flight feathers (visible in flight)",
    "Yellow billed cuckoo is a bird with tail that is very long, brown with bold large white spots on the underside of the outer tail feathers",
    "Yellow billed cuckoo is a bird with legs that are short, gray, with two toes forward and two back",
    "A photo of a yellow billed cuckoo, a slim long-tailed cuckoo with warm brown upperparts, white belly, rusty primaries flashing in flight, and a yellow lower mandible",
]
CUB_CLAUDE["yellow_breasted_chat"] = [
    "Yellow breasted chat is a bird with beak that is heavy, thicker than warbler bills, dark gray-black",
    "Yellow breasted chat is a bird with head that is olive-gray with bold white spectacles around the dark eye and a black face mask in males (lighter in females)",
    "Yellow breasted chat is a bird with body that is olive-green above, brilliant lemon-yellow throat and chest, white belly",
    "Yellow breasted chat is a bird with wings that are plain olive without obvious wing bars",
    "Yellow breasted chat is a bird with tail that is long, plain olive",
    "Yellow breasted chat is a bird with legs that are blue-gray, sturdy",
    "A photo of a yellow breasted chat, a large warbler-like songbird with bold white spectacles, brilliant yellow throat and chest, white belly, and a heavy bill (more like a tanager than a warbler)",
]
CUB_CLAUDE["yellow_headed_blackbird"] = [
    "Yellow headed blackbird is a bird with beak that is conical, sharply pointed, dark gray-black",
    "Yellow headed blackbird is a bird with head that is brilliant golden-yellow head, throat, and upper breast in adult males (with a small black mask through the eye), dull dingy-yellow in females",
    "Yellow headed blackbird is a bird with body that is jet-black overall in males contrasting with the yellow head, dingy gray-brown with a buffy-yellow throat in females",
    "Yellow headed blackbird is a bird with wings that are jet-black in males with a striking large white wing patch, brown in females without wing patch",
    "Yellow headed blackbird is a bird with tail that is medium, jet-black in males",
    "Yellow headed blackbird is a bird with legs that are dark, sturdy",
    "A photo of a yellow headed blackbird, a stocky black blackbird where adult males have a striking brilliant golden-yellow head and breast contrasting with white wing patches",
]
CUB_CLAUDE["yellow_throated_vireo"] = [
    "Yellow throated vireo is a bird with beak that is short, thick at the base, slightly hooked at the tip, dark gray with a paler lower mandible",
    "Yellow throated vireo is a bird with head that is olive-green crown, bright yellow throat and chest, prominent yellow spectacles formed by eye-ring and supraloral stripe",
    "Yellow throated vireo is a bird with body that is olive-green upperparts transitioning to gray rump, white belly with bright yellow upper breast and flanks",
    "Yellow throated vireo is a bird with wings that are dark gray-brown with two clean white wing bars, edges of flight feathers tinged olive",
    "Yellow throated vireo is a bird with tail that is medium length, square-tipped, dark gray with subtle white outer edges",
    "Yellow throated vireo is a bird with legs that are blue-gray, sturdy, medium length, ending in dark zygodactyl claws suited for foliage gleaning",
    "A photo of a yellow throated vireo, a small olive-and-yellow songbird with bold yellow spectacles and two white wing bars",
]
CUB_CLAUDE["yellow_warbler"] = [
    "Yellow warbler is a bird with beak that is small, slender, sharply pointed, dark gray",
    "Yellow warbler is a bird with head that is uniformly bright lemon-yellow with a dark eye, no contrasting face pattern",
    "Yellow warbler is a bird with body that is bright lemon-yellow overall, with rich rusty-chestnut breast and flank streaks in adult males (faint in females)",
    "Yellow warbler is a bird with wings that are yellow-edged dark with no obvious wing bars (yellow primary edges create a faint yellow look)",
    "Yellow warbler is a bird with tail that is short, with conspicuous yellow inner edges and yellow tail spots (unique among warblers)",
    "Yellow warbler is a bird with legs that are pinkish-buff, slender",
    "A photo of a yellow warbler, a small uniformly bright lemon-yellow warbler where adult males show rich rusty-chestnut streaks running down the breast and flanks",
]


# ============== 验证 + 保存 ==============
if __name__ == "__main__":
    import torch, os

    # 完整性检查
    expected = 200
    assert len(CUB_CLAUDE) == expected, \
        f"Expected {expected} classes, got {len(CUB_CLAUDE)}"
    for k, v in CUB_CLAUDE.items():
        assert len(v) == 7, f"Class '{k}' has {len(v)} sentences (expected 7)"
        for i, s in enumerate(v):
            assert isinstance(s, str) and len(s) > 0, \
                f"Class '{k}' sentence [{i}] is empty or non-string"

    # 与原 GPT-4 文件键集对齐检查
    orig = torch.load("data/gpt4_data/cub.pt", weights_only=False)
    orig_keys = set(orig.keys())
    new_keys  = set(CUB_CLAUDE.keys())
    missing = orig_keys - new_keys
    extra   = new_keys - orig_keys
    if missing:
        print(f"[WARN] Missing {len(missing)} classes from original: {sorted(missing)[:10]}...")
    if extra:
        print(f"[WARN] Extra {len(extra)} classes not in original: {sorted(extra)[:10]}...")

    out_path = "data/gpt4_data/cub_claude.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(CUB_CLAUDE, out_path)

    print(f"\n[OK] Saved {len(CUB_CLAUDE)} classes × 7 sentences → {out_path}")
    print(f"     File size: {os.path.getsize(out_path)/1024:.1f} KB")
    print(f"\n     Sample (first class):")
    first_k = next(iter(CUB_CLAUDE))
    for i, s in enumerate(CUB_CLAUDE[first_k]):
        print(f"       [{i}] {s[:100]}...")
